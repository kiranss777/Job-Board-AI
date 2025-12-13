from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from config import Config
from openai import OpenAI
import streamlit as st
import logging
from urllib.parse import quote_plus

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8") if "seaborn-v0_8" in plt.style.available else plt.style.use("seaborn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnowflakeAgent:
    """Agent for querying H1-B data from Snowflake and generating insights"""

    def __init__(self):
        """Initialize Snowflake connection using SQLAlchemy and get table name"""
        self.table_name = Config.SNOWFLAKE_TABLE

        # Validate required config
        required_config = {
            "SNOWFLAKE_ACCOUNT": Config.SNOWFLAKE_ACCOUNT,
            "SNOWFLAKE_USER": Config.SNOWFLAKE_USER,
            "SNOWFLAKE_PASSWORD": Config.SNOWFLAKE_PASSWORD,
            "SNOWFLAKE_WAREHOUSE": Config.SNOWFLAKE_WAREHOUSE,
            "SNOWFLAKE_DATABASE": Config.SNOWFLAKE_DATABASE,
            "SNOWFLAKE_SCHEMA": Config.SNOWFLAKE_SCHEMA,
        }

        missing = [k for k, v in required_config.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required Snowflake configuration: {', '.join(missing)}. Please check your .env file."
            )

        try:
            # Build connection string for SQLAlchemy
            # Format: snowflake://{USER}:{PASSWORD}@{ACCOUNT}/{DATABASE}/{SCHEMA}?warehouse={WAREHOUSE}
            account = Config.SNOWFLAKE_ACCOUNT.strip()
            user = Config.SNOWFLAKE_USER
            password = Config.SNOWFLAKE_PASSWORD
            database = Config.SNOWFLAKE_DATABASE
            schema = Config.SNOWFLAKE_SCHEMA
            warehouse = Config.SNOWFLAKE_WAREHOUSE

            # URL encode password if it contains special characters
            password_encoded = quote_plus(password)

            connection_string = f"snowflake://{user}:{password_encoded}@{account}/{database}/{schema}?warehouse={warehouse}"

            # Add role if specified
            if Config.SNOWFLAKE_ROLE:
                connection_string += f"&role={Config.SNOWFLAKE_ROLE}"

            logger.debug(f"🔗 Creating Snowflake connection: {connection_string.replace(password_encoded, '***')}")

            # Create SQLAlchemy engine with connection pooling
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_size=1,
                max_overflow=0,
                connect_args={
                    "login_timeout": 30,
                    "network_timeout": 30,
                }
            )

            # Test connection with retry
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with self.engine.connect() as conn:
                        result = conn.execute(
                            text("SELECT CURRENT_ACCOUNT(), CURRENT_REGION(), CURRENT_ROLE()")
                        )
                        self.connection_info = result.fetchone()
                        logger.info(f"✅ Connected to Snowflake: Account={self.connection_info[0]}, Region={self.connection_info[1]}, Role={self.connection_info[2]}")
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Connection attempt {attempt + 1} failed, retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        raise

            self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

        except Exception as e:
            error_str = str(e)
            st.error("❌ **Failed to connect to Snowflake**")
            st.error(f"**Error:** {error_str}")

            # Get current IP for whitelisting instructions
            try:
                import requests
                current_ip = requests.get("https://api.ipify.org", timeout=5).text
            except:
                current_ip = "Unable to fetch (check https://api.ipify.org)"

            # Check if it's a network/whitelisting error
            if (
                "250001" in error_str
                or "could not connect" in error_str.lower()
                or "network" in error_str.lower()
            ):
                st.warning(
                    f"""
**🔒 Connection Issue - IP Whitelisting or Network Policy**

**Your IP Address: `{current_ip}`**

**If you just whitelisted your IP, please wait 1-2 minutes for the policy to propagate.**

**Verify the network policy is applied:**
```sql
-- Check if policy exists
SHOW NETWORK POLICIES;

-- Check if user has policy assigned
SHOW USERS LIKE '{Config.SNOWFLAKE_USER}';
SELECT "name", "network_policy" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
```

**If policy is not applied, run:**
```sql
CREATE OR REPLACE NETWORK POLICY ALLOW_CURRENT_IP
  ALLOWED_IP_LIST = ('0.0.0.0/0')
  BLOCKED_IP_LIST = ();

ALTER USER {Config.SNOWFLAKE_USER} SET NETWORK_POLICY = ALLOW_CURRENT_IP;
```

**Or allow only your specific IP:**
```sql
CREATE OR REPLACE NETWORK POLICY ALLOW_CURRENT_IP
  ALLOWED_IP_LIST = ('{current_ip}/32')
  BLOCKED_IP_LIST = ();

ALTER USER {Config.SNOWFLAKE_USER} SET NETWORK_POLICY = ALLOW_CURRENT_IP;
```

**After running the SQL, wait 1-2 minutes, then click "🔄 Retry Connection".**
                """
                )
            else:
                st.info(
                    "Please check your Snowflake credentials and connection parameters in the .env file."
                )
            raise

    @property
    def QUESTIONS(self):
        """Predefined questions with their SQL queries"""
        table = self.table_name
        return {
            "Top 10 H1-B Sponsoring Companies by Total Approvals": {
                "query": f"""
                    SELECT
                        EMPLOYER_PETITIONER_NAME,
                        SUM(TOTAL_APPROVALS) AS TOTAL_APPROVALS,
                        SUM(TOTAL_CASES) AS TOTAL_CASES,
                        ROUND(SUM(TOTAL_APPROVALS) * 100.0 / NULLIF(SUM(TOTAL_CASES),0), 2) AS APPROVAL_RATE
                    FROM {table}
                    WHERE FISCAL_YEAR >= 2020
                    GROUP BY EMPLOYER_PETITIONER_NAME
                    ORDER BY TOTAL_APPROVALS DESC
                    LIMIT 10
                """,
                "description": "Shows the companies that have sponsored the most H1-B visas",
            },
            "H1-B Approvals by Fiscal Year": {
                "query": f"""
                    SELECT
                        FISCAL_YEAR,
                        SUM(NEW_EMPLOYMENT_APPROVAL) as NEW_EMPLOYMENT,
                        SUM(CONTINUATION_APPROVAL) as CONTINUATION,
                        SUM(CHANGE_WITH_SAME_EMPLOYER_APPROVAL) as CHANGE_SAME_EMPLOYER,
                        SUM(NEW_CONCURRENT_APPROVAL) as NEW_CONCURRENT,
                        SUM(CHANGE_OF_EMPLOYER_APPROVAL) as CHANGE_EMPLOYER,
                        SUM(AMENDED_APPROVAL) as AMENDED,
                        SUM(TOTAL_APPROVALS) as TOTAL_APPROVALS
                    FROM {table}
                    WHERE FISCAL_YEAR >= 2020
                    GROUP BY FISCAL_YEAR
                    ORDER BY FISCAL_YEAR
                """,
                "description": "Trends in H1-B approvals across different fiscal years",
            },
            "Top Industries by H1-B Sponsorship": {
                "query": f"""
                    SELECT
                        INDUSTRY_NAICS_CODE,
                        SUM(TOTAL_APPROVALS) as TOTAL_APPROVALS,
                        SUM(TOTAL_CASES) as TOTAL_CASES,
                        COUNT(DISTINCT EMPLOYER_PETITIONER_NAME) as UNIQUE_EMPLOYERS,
                        ROUND(AVG(APPROVAL_RATE) * 100, 2) as AVG_APPROVAL_RATE
                    FROM {table}
                    WHERE FISCAL_YEAR >= 2020
                    GROUP BY INDUSTRY_NAICS_CODE
                    ORDER BY TOTAL_APPROVALS DESC
                    LIMIT 15
                """,
                "description": "Industries that sponsor the most H1-B visas",
            },
            "H1-B Approval Rates by State": {
                "query": f"""
                    SELECT
                        PETITIONER_STATE,
                        SUM(TOTAL_APPROVALS) as TOTAL_APPROVALS,
                        SUM(TOTAL_DENIALS) as TOTAL_DENIALS,
                        SUM(TOTAL_CASES) as TOTAL_CASES,
                        ROUND(SUM(TOTAL_APPROVALS) * 100.0 / NULLIF(SUM(TOTAL_CASES), 0), 2) as APPROVAL_RATE_PCT,
                        COUNT(DISTINCT EMPLOYER_PETITIONER_NAME) as UNIQUE_EMPLOYERS
                    FROM {table}
                    WHERE FISCAL_YEAR >= 2020 AND PETITIONER_STATE IS NOT NULL
                    GROUP BY PETITIONER_STATE
                    HAVING SUM(TOTAL_CASES) > 100
                    ORDER BY TOTAL_APPROVALS DESC
                    LIMIT 20
                """,
                "description": "H1-B approval rates and volumes by US state",
            },
            "Top Cities for H1-B Sponsorship": {
                "query": f"""
                    SELECT
                        PETITIONER_CITY,
                        PETITIONER_STATE,
                        SUM(TOTAL_APPROVALS) as TOTAL_APPROVALS,
                        COUNT(DISTINCT EMPLOYER_PETITIONER_NAME) as UNIQUE_EMPLOYERS,
                        ROUND(AVG(APPROVAL_RATE) * 100, 2) as AVG_APPROVAL_RATE
                    FROM {table}
                    WHERE FISCAL_YEAR >= 2020 AND PETITIONER_CITY IS NOT NULL
                    GROUP BY PETITIONER_CITY, PETITIONER_STATE
                    ORDER BY TOTAL_APPROVALS DESC
                    LIMIT 20
                """,
                "description": "Cities with the highest H1-B sponsorship activity",
            },
            "H1-B Approval vs Denial Trends": {
                "query": f"""
                    SELECT
                        FISCAL_YEAR,
                        SUM(NEW_EMPLOYMENT_APPROVAL) as NEW_APPROVED,
                        SUM(NEW_EMPLOYMENT_DENIAL) as NEW_DENIED,
                        SUM(CONTINUATION_APPROVAL) as CONTINUATION_APPROVED,
                        SUM(CONTINUATION_DENIAL) as CONTINUATION_DENIED,
                        SUM(TOTAL_APPROVALS) as TOTAL_APPROVED,
                        SUM(TOTAL_DENIALS) as TOTAL_DENIED,
                        ROUND(SUM(TOTAL_APPROVALS) * 100.0 / NULLIF(SUM(TOTAL_CASES), 0), 2) as OVERALL_APPROVAL_RATE
                    FROM {table}
                    WHERE FISCAL_YEAR >= 2020
                    GROUP BY FISCAL_YEAR
                    ORDER BY FISCAL_YEAR
                """,
                "description": "Comparison of approvals and denials over time",
            },
            "Employment Type Distribution": {
                "query": f"""
                    SELECT
                        FISCAL_YEAR,
                        SUM(NEW_EMPLOYMENT_APPROVAL) as NEW_EMPLOYMENT,
                        SUM(CONTINUATION_APPROVAL) as CONTINUATION,
                        SUM(CHANGE_WITH_SAME_EMPLOYER_APPROVAL) as CHANGE_SAME_EMPLOYER,
                        SUM(NEW_CONCURRENT_APPROVAL) as NEW_CONCURRENT,
                        SUM(CHANGE_OF_EMPLOYER_APPROVAL) as CHANGE_EMPLOYER,
                        SUM(AMENDED_APPROVAL) as AMENDED
                    FROM {table}
                    WHERE FISCAL_YEAR >= 2020
                    GROUP BY FISCAL_YEAR
                    ORDER BY FISCAL_YEAR
                """,
                "description": "Distribution of different H1-B employment types",
            },
            "Top Companies by Approval Rate (Min 100 cases)": {
                "query": f"""
                    SELECT
                        EMPLOYER_PETITIONER_NAME,
                        SUM(TOTAL_APPROVALS) as TOTAL_APPROVALS,
                        SUM(TOTAL_DENIALS) as TOTAL_DENIALS,
                        SUM(TOTAL_CASES) as TOTAL_CASES,
                        ROUND(SUM(TOTAL_APPROVALS) * 100.0 / NULLIF(SUM(TOTAL_CASES), 0), 2) as APPROVAL_RATE_PCT
                    FROM {table}
                    WHERE FISCAL_YEAR >= 2020
                    GROUP BY EMPLOYER_PETITIONER_NAME
                    HAVING SUM(TOTAL_CASES) >= 100
                    ORDER BY APPROVAL_RATE_PCT DESC, TOTAL_CASES DESC
                    LIMIT 15
                """,
                "description": "Companies with the highest H1-B approval rates",
            },
            "Year-over-Year Growth in H1-B Sponsorship": {
                "query": f"""
                    SELECT
                        FISCAL_YEAR,
                        SUM(TOTAL_APPROVALS) as TOTAL_APPROVALS,
                        SUM(TOTAL_CASES) as TOTAL_CASES,
                        COUNT(DISTINCT EMPLOYER_PETITIONER_NAME) as UNIQUE_EMPLOYERS,
                        LAG(SUM(TOTAL_APPROVALS)) OVER (ORDER BY FISCAL_YEAR) as PREV_YEAR_APPROVALS,
                        ROUND(
                            (SUM(TOTAL_APPROVALS) - LAG(SUM(TOTAL_APPROVALS)) OVER (ORDER BY FISCAL_YEAR)) * 100.0 /
                            NULLIF(LAG(SUM(TOTAL_APPROVALS)) OVER (ORDER BY FISCAL_YEAR), 0),
                            2
                        ) as YOY_GROWTH_PCT
                    FROM {table}
                    WHERE FISCAL_YEAR >= 2020
                    GROUP BY FISCAL_YEAR
                    ORDER BY FISCAL_YEAR
                """,
                "description": "Year-over-year growth trends in H1-B sponsorship",
            },
            "Geographic Distribution of H1-B Sponsors": {
                "query": f"""
                    SELECT
                        PETITIONER_STATE,
                        COUNT(DISTINCT PETITIONER_CITY) as NUM_CITIES,
                        COUNT(DISTINCT EMPLOYER_PETITIONER_NAME) as NUM_EMPLOYERS,
                        SUM(TOTAL_APPROVALS) as TOTAL_APPROVALS,
                        SUM(TOTAL_CASES) as TOTAL_CASES
                    FROM {table}
                    WHERE FISCAL_YEAR >= 2020 AND PETITIONER_STATE IS NOT NULL
                    GROUP BY PETITIONER_STATE
                    HAVING SUM(TOTAL_CASES) > 50
                    ORDER BY TOTAL_APPROVALS DESC
                    LIMIT 25
                """,
                "description": "Geographic spread of H1-B sponsorship across US states",
            },
        }

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            logger.debug(f"📤 Running query: {query[:100]}...")
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                # Convert result to list of dictionaries
                rows = [dict(row._mapping) for row in result]
                logger.info(f"✅ Query returned {len(rows)} rows")
                return pd.DataFrame(rows)
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            logger.error(f"❌ Query failed: {str(e)}")
            raise

    def generate_summary(self, question: str, df: pd.DataFrame) -> str:
        """Generate AI summary of query results using GPT-4o-mini"""
        try:
            # Prepare data summary for LLM
            data_summary = f"""
            Question: {question}
            
            Data Summary:
            - Number of rows: {len(df)}
            - Columns: {', '.join(df.columns.tolist())}
            - First few rows:
            {df.head(10).to_string()}
            
            Statistical Summary:
            {df.describe().to_string() if len(df.select_dtypes(include=['number']).columns) > 0 else 'No numerical columns'}
            """

            prompt = f"""You are a data analyst specializing in H1-B visa sponsorship data. 
            Analyze the following query results and provide a comprehensive, insightful summary.
            
            {data_summary}
            
            Please provide:
            1. Key findings and insights
            2. Notable trends or patterns
            3. Important statistics
            4. Any interesting observations
            
            Keep the summary concise but informative (3-5 paragraphs)."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data analyst specializing in immigration and employment visa data.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"Error generating summary: {str(e)}")
            return "Summary generation failed. Please review the data manually."

    def _create_fiscal_year_approvals_viz(self, df, fiscal_year_col, numeric_cols):
        """Create visualizations for H1-B Approvals by Fiscal Year"""
        figures = []
        year_df = df.sort_values(fiscal_year_col).copy()
        
        # Graph 1: Total Approvals
        try:
            total_col = [col for col in numeric_cols if col.upper() == "TOTAL_APPROVALS"][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=year_df, x=fiscal_year_col, y=total_col, marker="o", 
                       markersize=10, linewidth=3, color="#2E86AB", ax=ax)
            ax.fill_between(year_df[fiscal_year_col], year_df[total_col], alpha=0.3, color="#2E86AB")
            ax.set_title("Fiscal Year vs Total Approvals", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(year_df[fiscal_year_col])
            for x, y in zip(year_df[fiscal_year_col], year_df[total_col]):
                ax.annotate(f'{y:,.0f}', (x, y), textcoords="offset points", xytext=(0,10), 
                          ha='center', fontsize=9, fontweight="bold")
            plt.tight_layout()
            figures.append((fig, "Graph 1: Fiscal Year vs Total Approvals"))
        except: pass
        
        # Graph 2: New Employment
        try:
            new_emp_col = [col for col in numeric_cols if "NEW_EMPLOYMENT" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=year_df, x=fiscal_year_col, y=new_emp_col, ax=ax, 
                      palette="viridis", hue=fiscal_year_col, legend=False)
            ax.set_title(f"Fiscal Year vs {new_emp_col.replace('_', ' ').title()}", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Approvals", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, f"Graph 2: Fiscal Year vs {new_emp_col.replace('_', ' ').title()}"))
        except: pass
        
        # Graph 3: Continuation
        try:
            cont_col = [col for col in numeric_cols if "CONTINUATION" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=year_df, x=fiscal_year_col, y=cont_col, marker="s", 
                       markersize=8, linewidth=2.5, color="#FF6B35", ax=ax)
            ax.fill_between(year_df[fiscal_year_col], year_df[cont_col], alpha=0.2, color="#FF6B35")
            ax.set_title(f"Fiscal Year vs {cont_col.replace('_', ' ').title()}", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Approvals", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(year_df[fiscal_year_col])
            plt.tight_layout()
            figures.append((fig, f"Graph 3: Fiscal Year vs {cont_col.replace('_', ' ').title()}"))
        except: pass
        
        # Graph 4: Change Employer
        try:
            change_col = [col for col in numeric_cols if "CHANGE" in col.upper() and "EMPLOYER" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=year_df, x=fiscal_year_col, y=change_col, ax=ax, 
                      palette="Set2", hue=fiscal_year_col, legend=False)
            ax.set_title(f"Fiscal Year vs {change_col.replace('_', ' ').title()}", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Approvals", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, f"Graph 4: Fiscal Year vs {change_col.replace('_', ' ').title()}"))
        except: pass
        
        return figures

    def _create_approval_denial_trends_viz(self, df, fiscal_year_col, numeric_cols):
        """Create visualizations for Approval vs Denial Trends"""
        figures = []
        year_df = df.sort_values(fiscal_year_col).copy()
        
        # Graph 1: Total Approved vs Total Denied
        try:
            approved_col = [col for col in numeric_cols if "APPROVED" in col.upper() and "TOTAL" in col.upper()][0]
            denied_col = [col for col in numeric_cols if "DENIED" in col.upper() and "TOTAL" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=year_df, x=fiscal_year_col, y=approved_col, marker="o", label="Approved", linewidth=2.5, ax=ax)
            sns.lineplot(data=year_df, x=fiscal_year_col, y=denied_col, marker="s", label="Denied", linewidth=2.5, ax=ax)
            ax.set_title("Fiscal Year vs Total Approved vs Total Denied", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Cases", fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(year_df[fiscal_year_col])
            plt.tight_layout()
            figures.append((fig, "Graph 1: Approved vs Denied Trends"))
        except: pass
        
        # Graph 2: Approval Rate Over Time
        try:
            rate_col = [col for col in numeric_cols if "APPROVAL_RATE" in col.upper() or "RATE" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=year_df, x=fiscal_year_col, y=rate_col, ax=ax, palette="RdYlGn", hue=fiscal_year_col, legend=False)
            ax.set_title("Fiscal Year vs Overall Approval Rate (%)", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Approval Rate (%)", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 2: Approval Rate Trend"))
        except: pass
        
        # Graph 3: Stacked Bar Chart
        try:
            approved_col = [col for col in numeric_cols if "APPROVED" in col.upper() and "TOTAL" in col.upper()][0]
            denied_col = [col for col in numeric_cols if "DENIED" in col.upper() and "TOTAL" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            x = year_df[fiscal_year_col]
            width = 0.6
            ax.bar(x, year_df[approved_col], width, label='Approved', color='#2E86AB', alpha=0.8)
            ax.bar(x, year_df[denied_col], width, bottom=year_df[approved_col], label='Denied', color='#FF6B35', alpha=0.8)
            ax.set_title("Fiscal Year vs Approved vs Denied (Stacked)", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Cases", fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 3: Stacked Approved vs Denied"))
        except: pass
        
        # Graph 4: New Employment Approved vs Denied
        try:
            new_app = [col for col in numeric_cols if "NEW" in col.upper() and "APPROVED" in col.upper()][0]
            new_den = [col for col in numeric_cols if "NEW" in col.upper() and "DENIED" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=year_df, x=fiscal_year_col, y=new_app, marker="o", label="New Approved", linewidth=2.5, ax=ax)
            sns.lineplot(data=year_df, x=fiscal_year_col, y=new_den, marker="s", label="New Denied", linewidth=2.5, ax=ax)
            ax.set_title("Fiscal Year vs New Employment: Approved vs Denied", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Cases", fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(year_df[fiscal_year_col])
            plt.tight_layout()
            figures.append((fig, "Graph 4: New Employment Approved vs Denied"))
        except: pass
        
        return figures

    def _create_employment_type_distribution_viz(self, df, fiscal_year_col, numeric_cols):
        """Create visualizations for Employment Type Distribution"""
        figures = []
        year_df = df.sort_values(fiscal_year_col).copy()
        
        # Graph 1: Stacked Area Chart
        try:
            fig, ax = plt.subplots(figsize=(14, 7))
            colors = sns.color_palette("Set2", len(numeric_cols))
            ax.stackplot(year_df[fiscal_year_col], *[year_df[col] for col in numeric_cols],
                        labels=[col.replace("_", " ").title() for col in numeric_cols],
                        alpha=0.7, colors=colors, edgecolor="white", linewidth=1.5)
            ax.set_title("Fiscal Year vs Employment Types (Stacked Area)", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Approvals", fontsize=13, fontweight="bold")
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=2)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(year_df[fiscal_year_col])
            plt.tight_layout()
            figures.append((fig, "Graph 1: Employment Types Stacked Area"))
        except: pass
        
        # Graph 2: Grouped Bar Chart
        try:
            fig, ax = plt.subplots(figsize=(14, 7))
            plot_df = year_df.melt(id_vars=[fiscal_year_col], value_vars=numeric_cols, 
                                  var_name="Employment Type", value_name="Count")
            plot_df["Employment Type"] = plot_df["Employment Type"].str.replace("_", " ").str.title()
            sns.barplot(data=plot_df, x=fiscal_year_col, y="Count", hue="Employment Type", ax=ax, palette="Set2")
            ax.set_title("Fiscal Year vs Employment Types (Grouped Bar)", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Approvals", fontsize=13, fontweight="bold")
            ax.legend(title="Employment Type", loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 2: Employment Types Grouped Bar"))
        except: pass
        
        # Graph 3: Pie Chart for Latest Year
        try:
            latest_year = year_df[fiscal_year_col].max()
            latest_data = year_df[year_df[fiscal_year_col] == latest_year].iloc[0]
            fig, ax = plt.subplots(figsize=(10, 8))
            values = [latest_data[col] for col in numeric_cols]
            labels = [col.replace("_", " ").title() for col in numeric_cols]
            colors_pie = sns.color_palette("Set2", len(numeric_cols))
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_pie)
            ax.set_title(f"Employment Type Distribution for {latest_year}", fontsize=16, fontweight="bold", pad=20)
            plt.tight_layout()
            figures.append((fig, f"Graph 3: Distribution for {latest_year}"))
        except: pass
        
        # Graph 4: Line Chart for All Types
        try:
            fig, ax = plt.subplots(figsize=(14, 7))
            for col in numeric_cols:
                sns.lineplot(data=year_df, x=fiscal_year_col, y=col, marker="o", 
                           label=col.replace("_", " ").title(), linewidth=2, ax=ax)
            ax.set_title("Fiscal Year vs All Employment Types (Line Chart)", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Approvals", fontsize=13, fontweight="bold")
            ax.legend(loc="best", fontsize=9, framealpha=0.9, ncol=2)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(year_df[fiscal_year_col])
            plt.tight_layout()
            figures.append((fig, "Graph 4: All Employment Types Line Chart"))
        except: pass
        
        return figures

    def _create_yoy_growth_viz(self, df, fiscal_year_col, numeric_cols):
        """Create visualizations for Year-over-Year Growth"""
        figures = []
        year_df = df.sort_values(fiscal_year_col).copy()
        
        # Graph 1: YoY Growth Percentage
        try:
            yoy_col = [col for col in numeric_cols if "YOY" in col.upper() or "GROWTH" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['green' if x > 0 else 'red' for x in year_df[yoy_col]]
            sns.barplot(data=year_df, x=fiscal_year_col, y=yoy_col, ax=ax, palette=colors, hue=fiscal_year_col, legend=False)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_title("Fiscal Year vs Year-over-Year Growth (%)", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("YoY Growth (%)", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 1: YoY Growth Percentage"))
        except: pass
        
        # Graph 2: Total Approvals Trend
        try:
            total_col = [col for col in numeric_cols if "TOTAL_APPROVALS" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=year_df, x=fiscal_year_col, y=total_col, marker="o", 
                       markersize=10, linewidth=3, color="#2E86AB", ax=ax)
            ax.fill_between(year_df[fiscal_year_col], year_df[total_col], alpha=0.3, color="#2E86AB")
            ax.set_title("Fiscal Year vs Total Approvals", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(year_df[fiscal_year_col])
            plt.tight_layout()
            figures.append((fig, "Graph 2: Total Approvals Trend"))
        except: pass
        
        # Graph 3: Unique Employers
        try:
            emp_col = [col for col in numeric_cols if "EMPLOYERS" in col.upper() or "UNIQUE" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=year_df, x=fiscal_year_col, y=emp_col, ax=ax, 
                      palette="viridis", hue=fiscal_year_col, legend=False)
            ax.set_title(f"Fiscal Year vs {emp_col.replace('_', ' ').title()}", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Employers", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, f"Graph 3: {emp_col.replace('_', ' ').title()}"))
        except: pass
        
        # Graph 4: Comparison with Previous Year
        try:
            total_col = [col for col in numeric_cols if "TOTAL_APPROVALS" in col.upper()][0]
            prev_col = [col for col in numeric_cols if "PREV" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=year_df, x=fiscal_year_col, y=total_col, marker="o", label="Current Year", linewidth=2.5, ax=ax)
            sns.lineplot(data=year_df, x=fiscal_year_col, y=prev_col, marker="s", label="Previous Year", linewidth=2.5, ax=ax)
            ax.set_title("Fiscal Year vs Current vs Previous Year Approvals", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Approvals", fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xticks(year_df[fiscal_year_col])
            plt.tight_layout()
            figures.append((fig, "Graph 4: Current vs Previous Year"))
        except: pass
        
        return figures

    def _create_top_companies_viz(self, df, categorical_cols, numeric_cols):
        """Create visualizations for Top Companies"""
        figures = []
        company_col = categorical_cols[0] if categorical_cols else None
        total_col = [col for col in numeric_cols if "TOTAL_APPROVALS" in col.upper()][0] if numeric_cols else None
        
        if not company_col or not total_col:
            return figures
        
        # Graph 1: Horizontal Bar Chart
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_df = df.nlargest(10, total_col)
            sns.barplot(data=top_df, x=total_col, y=company_col, ax=ax, palette="viridis", hue=company_col, legend=False)
            ax.set_title("Top 10 Companies by Total Approvals", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.set_ylabel("Company Name", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 1: Top 10 Companies Bar Chart"))
        except: pass
        
        # Graph 2: Pie Chart
        try:
            top_df = df.nlargest(8, total_col)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(top_df[total_col], labels=top_df[company_col], autopct='%1.1f%%', startangle=90)
            ax.set_title("Top 8 Companies Distribution", fontsize=16, fontweight="bold", pad=20)
            plt.tight_layout()
            figures.append((fig, "Graph 2: Top Companies Pie Chart"))
        except: pass
        
        # Graph 3: Scatter Plot (Total Approvals vs Approval Rate)
        try:
            rate_col = [col for col in numeric_cols if "APPROVAL_RATE" in col.upper() or "RATE" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(data=df, x=total_col, y=rate_col, size=total_col, sizes=(50, 500), alpha=0.6, ax=ax)
            ax.set_title("Total Approvals vs Approval Rate", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.set_ylabel("Approval Rate", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 3: Approvals vs Approval Rate Scatter"))
        except: pass
        
        # Graph 4: Total Cases Comparison
        try:
            cases_col = [col for col in numeric_cols if "TOTAL_CASES" in col.upper()][0]
            top_df = df.nlargest(10, total_col)
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(top_df))
            width = 0.35
            ax.bar(x - width/2, top_df[total_col], width, label='Approvals', alpha=0.8)
            ax.bar(x + width/2, top_df[cases_col], width, label='Total Cases', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in top_df[company_col]], rotation=45, ha='right')
            ax.set_title("Top 10 Companies: Approvals vs Total Cases", fontsize=16, fontweight="bold", pad=20)
            ax.set_ylabel("Count", fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 4: Approvals vs Cases Comparison"))
        except: pass
        
        return figures

    def _create_companies_approval_rate_viz(self, df, categorical_cols, numeric_cols):
        """Create visualizations for Companies by Approval Rate"""
        figures = []
        company_col = categorical_cols[0] if categorical_cols else None
        rate_col = [col for col in numeric_cols if "APPROVAL_RATE" in col.upper() or "RATE" in col.upper()][0] if numeric_cols else None
        
        if not company_col or not rate_col:
            return figures
        
        # Graph 1: Horizontal Bar Chart by Approval Rate
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_df = df.nlargest(15, rate_col)
            sns.barplot(data=top_df, x=rate_col, y=company_col, ax=ax, palette="RdYlGn", hue=company_col, legend=False)
            ax.set_title("Top Companies by Approval Rate (%)", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Approval Rate (%)", fontsize=13, fontweight="bold")
            ax.set_ylabel("Company Name", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 1: Top Companies by Approval Rate"))
        except: pass
        
        # Graph 2: Approval Rate vs Total Cases
        try:
            cases_col = [col for col in numeric_cols if "TOTAL_CASES" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(data=df, x=cases_col, y=rate_col, size=cases_col, sizes=(50, 500), alpha=0.6, ax=ax)
            ax.set_title("Approval Rate vs Total Cases", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Cases", fontsize=13, fontweight="bold")
            ax.set_ylabel("Approval Rate (%)", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 2: Approval Rate vs Cases Scatter"))
        except: pass
        
        # Graph 3: Approvals vs Denials
        try:
            app_col = [col for col in numeric_cols if "TOTAL_APPROVALS" in col.upper()][0]
            den_col = [col for col in numeric_cols if "TOTAL_DENIALS" in col.upper()][0]
            top_df = df.nlargest(10, rate_col)
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(top_df))
            width = 0.35
            ax.bar(x - width/2, top_df[app_col], width, label='Approvals', color='green', alpha=0.8)
            ax.bar(x + width/2, top_df[den_col], width, label='Denials', color='red', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in top_df[company_col]], rotation=45, ha='right')
            ax.set_title("Top 10 Companies: Approvals vs Denials", fontsize=16, fontweight="bold", pad=20)
            ax.set_ylabel("Count", fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 3: Approvals vs Denials"))
        except: pass
        
        # Graph 4: Approval Rate Distribution
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x=rate_col, bins=20, kde=True, ax=ax, color="#2E86AB")
            ax.set_title("Approval Rate Distribution", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Approval Rate (%)", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Companies", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 4: Approval Rate Distribution"))
        except: pass
        
        return figures

    def _create_top_industries_viz(self, df, categorical_cols, numeric_cols):
        """Create visualizations for Top Industries"""
        figures = []
        industry_col = categorical_cols[0] if categorical_cols else None
        total_col = [col for col in numeric_cols if "TOTAL_APPROVALS" in col.upper()][0] if numeric_cols else None
        
        if not industry_col or not total_col:
            return figures
        
        # Graph 1: Horizontal Bar Chart
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(data=df, x=total_col, y=industry_col, ax=ax, palette="viridis", hue=industry_col, legend=False)
            ax.set_title("Top Industries by Total Approvals", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.set_ylabel("Industry Code", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 1: Top Industries Bar Chart"))
        except: pass
        
        # Graph 2: Scatter Plot (Total Approvals vs Unique Employers)
        try:
            emp_col = [col for col in numeric_cols if "EMPLOYERS" in col.upper() or "UNIQUE" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(data=df, x=total_col, y=emp_col, size=total_col, sizes=(50, 500), alpha=0.6, ax=ax)
            ax.set_title("Total Approvals vs Unique Employers", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.set_ylabel("Unique Employers", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 2: Approvals vs Employers Scatter"))
        except: pass
        
        # Graph 3: Average Approval Rate
        try:
            rate_col = [col for col in numeric_cols if "APPROVAL_RATE" in col.upper() or "RATE" in col.upper()][0]
            top_df = df.nlargest(15, total_col)
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=top_df, x=rate_col, y=industry_col, ax=ax, palette="RdYlGn", hue=industry_col, legend=False)
            ax.set_title("Top Industries by Average Approval Rate (%)", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Average Approval Rate (%)", fontsize=13, fontweight="bold")
            ax.set_ylabel("Industry Code", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 3: Industries by Approval Rate"))
        except: pass
        
        # Graph 4: Total Cases Comparison
        try:
            cases_col = [col for col in numeric_cols if "TOTAL_CASES" in col.upper()][0]
            top_df = df.nlargest(10, total_col)
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(top_df))
            width = 0.35
            ax.bar(x - width/2, top_df[total_col], width, label='Approvals', alpha=0.8)
            ax.bar(x + width/2, top_df[cases_col], width, label='Total Cases', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(top_df[industry_col], rotation=45, ha='right')
            ax.set_title("Top 10 Industries: Approvals vs Total Cases", fontsize=16, fontweight="bold", pad=20)
            ax.set_ylabel("Count", fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 4: Approvals vs Cases"))
        except: pass
        
        return figures

    def _create_state_approval_rates_viz(self, df, categorical_cols, numeric_cols):
        """Create visualizations for State Approval Rates"""
        figures = []
        state_col = categorical_cols[0] if categorical_cols else None
        rate_col = [col for col in numeric_cols if "APPROVAL_RATE" in col.upper() or "RATE" in col.upper()][0] if numeric_cols else None
        
        if not state_col or not rate_col:
            return figures
        
        # Graph 1: Horizontal Bar Chart by Approval Rate
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            top_df = df.nlargest(20, rate_col)
            sns.barplot(data=top_df, x=rate_col, y=state_col, ax=ax, palette="RdYlGn", hue=state_col, legend=False)
            ax.set_title("Top States by Approval Rate (%)", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Approval Rate (%)", fontsize=13, fontweight="bold")
            ax.set_ylabel("State", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 1: States by Approval Rate"))
        except: pass
        
        # Graph 2: Total Approvals by State
        try:
            total_col = [col for col in numeric_cols if "TOTAL_APPROVALS" in col.upper()][0]
            top_df = df.nlargest(20, total_col)
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.barplot(data=top_df, x=total_col, y=state_col, ax=ax, palette="viridis", hue=state_col, legend=False)
            ax.set_title("Top States by Total Approvals", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.set_ylabel("State", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 2: States by Total Approvals"))
        except: pass
        
        # Graph 3: Scatter Plot (Approval Rate vs Total Cases)
        try:
            cases_col = [col for col in numeric_cols if "TOTAL_CASES" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(data=df, x=cases_col, y=rate_col, size=total_col, sizes=(50, 500), alpha=0.6, ax=ax)
            ax.set_title("Approval Rate vs Total Cases by State", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Cases", fontsize=13, fontweight="bold")
            ax.set_ylabel("Approval Rate (%)", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 3: Approval Rate vs Cases Scatter"))
        except: pass
        
        # Graph 4: Approvals vs Denials
        try:
            total_col = [col for col in numeric_cols if "TOTAL_APPROVALS" in col.upper()][0]
            den_col = [col for col in numeric_cols if "TOTAL_DENIALS" in col.upper()][0]
            top_df = df.nlargest(15, total_col)
            fig, ax = plt.subplots(figsize=(12, 8))
            x = np.arange(len(top_df))
            width = 0.35
            ax.bar(x - width/2, top_df[total_col], width, label='Approvals', color='green', alpha=0.8)
            ax.bar(x + width/2, top_df[den_col], width, label='Denials', color='red', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(top_df[state_col], rotation=45, ha='right')
            ax.set_title("Top 15 States: Approvals vs Denials", fontsize=16, fontweight="bold", pad=20)
            ax.set_ylabel("Count", fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 4: States Approvals vs Denials"))
        except: pass
        
        return figures

    def _create_top_cities_viz(self, df, categorical_cols, numeric_cols):
        """Create visualizations for Top Cities"""
        figures = []
        city_col = [col for col in categorical_cols if "CITY" in col.upper()][0] if categorical_cols else None
        total_col = [col for col in numeric_cols if "TOTAL_APPROVALS" in col.upper()][0] if numeric_cols else None
        
        if not city_col or not total_col:
            return figures
        
        # Graph 1: Horizontal Bar Chart
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            top_df = df.nlargest(20, total_col)
            sns.barplot(data=top_df, x=total_col, y=city_col, ax=ax, palette="viridis", hue=city_col, legend=False)
            ax.set_title("Top 20 Cities by Total Approvals", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.set_ylabel("City", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 1: Top Cities Bar Chart"))
        except: pass
        
        # Graph 2: Unique Employers by City
        try:
            emp_col = [col for col in numeric_cols if "EMPLOYERS" in col.upper() or "UNIQUE" in col.upper()][0]
            top_df = df.nlargest(20, total_col)
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.barplot(data=top_df, x=emp_col, y=city_col, ax=ax, palette="Set2", hue=city_col, legend=False)
            ax.set_title("Top 20 Cities by Unique Employers", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Unique Employers", fontsize=13, fontweight="bold")
            ax.set_ylabel("City", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 2: Cities by Unique Employers"))
        except: pass
        
        # Graph 3: Scatter Plot (Total Approvals vs Average Approval Rate)
        try:
            rate_col = [col for col in numeric_cols if "APPROVAL_RATE" in col.upper() or "RATE" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(data=df, x=total_col, y=rate_col, size=total_col, sizes=(50, 500), alpha=0.6, ax=ax)
            ax.set_title("Total Approvals vs Average Approval Rate by City", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.set_ylabel("Average Approval Rate (%)", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 3: Approvals vs Approval Rate Scatter"))
        except: pass
        
        # Graph 4: Grouped Bar Chart (if state column exists)
        try:
            state_col = [col for col in categorical_cols if "STATE" in col.upper()][0]
            top_df = df.nlargest(15, total_col)
            fig, ax = plt.subplots(figsize=(14, 8))
            plot_df = top_df.melt(id_vars=[city_col, state_col], value_vars=[total_col], 
                                 var_name="Metric", value_name="Count")
            sns.barplot(data=top_df, x=city_col, y=total_col, hue=state_col, ax=ax, palette="Set2")
            ax.set_title("Top 15 Cities: Total Approvals by State", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("City", fontsize=13, fontweight="bold")
            ax.set_ylabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title="State", fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, axis="y", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 4: Cities by State Grouped"))
        except: pass
        
        return figures

    def _create_geographic_distribution_viz(self, df, categorical_cols, numeric_cols):
        """Create visualizations for Geographic Distribution"""
        figures = []
        state_col = categorical_cols[0] if categorical_cols else None
        total_col = [col for col in numeric_cols if "TOTAL_APPROVALS" in col.upper()][0] if numeric_cols else None
        
        if not state_col or not total_col:
            return figures
        
        # Graph 1: Horizontal Bar Chart by Total Approvals
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            top_df = df.nlargest(25, total_col)
            sns.barplot(data=top_df, x=total_col, y=state_col, ax=ax, palette="viridis", hue=state_col, legend=False)
            ax.set_title("Geographic Distribution: Top States by Total Approvals", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.set_ylabel("State", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 1: States by Total Approvals"))
        except: pass
        
        # Graph 2: Number of Cities by State
        try:
            cities_col = [col for col in numeric_cols if "CITIES" in col.upper() or "NUM_CITIES" in col.upper()][0]
            top_df = df.nlargest(25, total_col)
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.barplot(data=top_df, x=cities_col, y=state_col, ax=ax, palette="Set2", hue=state_col, legend=False)
            ax.set_title("Number of Cities by State", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Number of Cities", fontsize=13, fontweight="bold")
            ax.set_ylabel("State", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 2: Cities Count by State"))
        except: pass
        
        # Graph 3: Unique Employers by State
        try:
            emp_col = [col for col in numeric_cols if "EMPLOYERS" in col.upper() or "NUM_EMPLOYERS" in col.upper()][0]
            top_df = df.nlargest(25, total_col)
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.barplot(data=top_df, x=emp_col, y=state_col, ax=ax, palette="RdYlGn", hue=state_col, legend=False)
            ax.set_title("Number of Unique Employers by State", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Number of Employers", fontsize=13, fontweight="bold")
            ax.set_ylabel("State", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x", linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 3: Employers by State"))
        except: pass
        
        # Graph 4: Scatter Plot (Total Approvals vs Number of Cities)
        try:
            cities_col = [col for col in numeric_cols if "CITIES" in col.upper() or "NUM_CITIES" in col.upper()][0]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(data=df, x=total_col, y=cities_col, size=total_col, sizes=(50, 500), alpha=0.6, ax=ax)
            ax.set_title("Total Approvals vs Number of Cities by State", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Total Approvals", fontsize=13, fontweight="bold")
            ax.set_ylabel("Number of Cities", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            plt.tight_layout()
            figures.append((fig, "Graph 4: Approvals vs Cities Scatter"))
        except: pass
        
        return figures

    def create_visualizations(
        self, df: pd.DataFrame, question: str
    ) -> List[Tuple[plt.Figure, str]]:
        """Create multiple visualizations based on the data and question type"""
        figures = []

        if df.empty:
            logger.warning("DataFrame is empty, cannot create visualizations")
            return figures

        # Get numeric columns (exclude FISCAL_YEAR for separate handling - case insensitive)
        all_numeric = df.select_dtypes(include=["number"]).columns.tolist()
        # Find fiscal year column (case insensitive)
        fiscal_year_col = None
        for col in df.columns:
            if col.upper() == "FISCAL_YEAR":
                fiscal_year_col = col
                break
        
        # Exclude fiscal year from numeric cols
        numeric_cols = [col for col in all_numeric if col.upper() != "FISCAL_YEAR"]
        categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        has_fiscal_year = fiscal_year_col is not None

        logger.info(f"Creating visualizations for question: {question}")
        logger.info(f"has_fiscal_year: {has_fiscal_year}, fiscal_year_col: {fiscal_year_col}, numeric_cols: {numeric_cols}")

        # Route to specific visualization functions based on question type
        question_lower = question.lower()
        
        if "approvals by fiscal year" in question_lower and has_fiscal_year:
            figures = self._create_fiscal_year_approvals_viz(df, fiscal_year_col, numeric_cols)
        elif "approval vs denial trends" in question_lower and has_fiscal_year:
            figures = self._create_approval_denial_trends_viz(df, fiscal_year_col, numeric_cols)
        elif "employment type distribution" in question_lower and has_fiscal_year:
            figures = self._create_employment_type_distribution_viz(df, fiscal_year_col, numeric_cols)
        elif "year-over-year growth" in question_lower and has_fiscal_year:
            figures = self._create_yoy_growth_viz(df, fiscal_year_col, numeric_cols)
        elif "top companies" in question_lower and "approval rate" in question_lower:
            figures = self._create_companies_approval_rate_viz(df, categorical_cols, numeric_cols)
        elif "top 10" in question_lower and "companies" in question_lower:
            figures = self._create_top_companies_viz(df, categorical_cols, numeric_cols)
        elif "top industries" in question_lower:
            figures = self._create_top_industries_viz(df, categorical_cols, numeric_cols)
        elif "approval rates by state" in question_lower:
            figures = self._create_state_approval_rates_viz(df, categorical_cols, numeric_cols)
        elif "top cities" in question_lower:
            figures = self._create_top_cities_viz(df, categorical_cols, numeric_cols)
        elif "geographic distribution" in question_lower:
            figures = self._create_geographic_distribution_viz(df, categorical_cols, numeric_cols)
        elif has_fiscal_year and len(numeric_cols) > 0:
            year_df = df.sort_values(fiscal_year_col).copy()
            
            # Get all columns including TOTAL_APPROVALS for individual comparisons
            all_cols_to_plot = numeric_cols.copy()
            # Prioritize TOTAL_APPROVALS if it exists
            if "TOTAL_APPROVALS" in [col.upper() for col in numeric_cols]:
                total_col = [col for col in numeric_cols if col.upper() == "TOTAL_APPROVALS"][0]
                all_cols_to_plot = [total_col] + [col for col in numeric_cols if col.upper() != "TOTAL_APPROVALS"]
            
            logger.info(f"Creating visualizations with fiscal_year_col: {fiscal_year_col}, all_cols_to_plot: {all_cols_to_plot}")
            
            # Visualization 1: Fiscal Year vs Total Approvals
            try:
                total_col = None
                for col in numeric_cols:
                    if col.upper() == "TOTAL_APPROVALS":
                        total_col = col
                        break
                
                if total_col:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(data=year_df, x=fiscal_year_col, y=total_col, marker="o", 
                               markersize=10, linewidth=3, color="#2E86AB", ax=ax)
                    ax.fill_between(year_df[fiscal_year_col], year_df[total_col], alpha=0.3, color="#2E86AB")
                    ax.set_title("Fiscal Year vs Total Approvals", fontsize=16, fontweight="bold", pad=20)
                    ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
                    ax.set_ylabel("Total Approvals", fontsize=13, fontweight="bold")
                    ax.grid(True, alpha=0.3, linestyle="--")
                    ax.set_xticks(year_df[fiscal_year_col])
                    # Add value labels on points
                    for x, y in zip(year_df[fiscal_year_col], year_df[total_col]):
                        ax.annotate(f'{y:,.0f}', (x, y), textcoords="offset points", xytext=(0,10), 
                                  ha='center', fontsize=9, fontweight="bold")
                    plt.tight_layout()
                    figures.append((fig, "Graph 1: Fiscal Year vs Total Approvals"))
                    logger.info("Created Graph 1: Total Approvals")
            except Exception as e:
                logger.error(f"Could not create total approvals chart: {str(e)}", exc_info=True)

            # Visualization 2: Fiscal Year vs New Employment
            try:
                new_emp_col = None
                for col in numeric_cols:
                    if "NEW_EMPLOYMENT" in col.upper() or "NEW EMPLOYMENT" in col.upper():
                        new_emp_col = col
                        break
                
                if new_emp_col:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(data=year_df, x=fiscal_year_col, y=new_emp_col, ax=ax, 
                              palette="viridis", hue=fiscal_year_col, legend=False)
                    col_title = new_emp_col.replace("_", " ").title()
                    ax.set_title(f"Fiscal Year vs {col_title}", fontsize=16, fontweight="bold", pad=20)
                    ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
                    ax.set_ylabel("Number of Approvals", fontsize=13, fontweight="bold")
                    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
                    plt.tight_layout()
                    figures.append((fig, f"Graph 2: Fiscal Year vs {col_title}"))
                    logger.info(f"Created Graph 2: {col_title}")
            except Exception as e:
                logger.warning(f"Could not create new employment chart: {str(e)}")

            # Visualization 3: Fiscal Year vs Continuation
            try:
                cont_col = None
                for col in numeric_cols:
                    if "CONTINUATION" in col.upper():
                        cont_col = col
                        break
                
                if cont_col:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(data=year_df, x=fiscal_year_col, y=cont_col, marker="s", 
                               markersize=8, linewidth=2.5, color="#FF6B35", ax=ax)
                    ax.fill_between(year_df[fiscal_year_col], year_df[cont_col], alpha=0.2, color="#FF6B35")
                    col_title = cont_col.replace("_", " ").title()
                    ax.set_title(f"Fiscal Year vs {col_title}", fontsize=16, fontweight="bold", pad=20)
                    ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
                    ax.set_ylabel("Number of Approvals", fontsize=13, fontweight="bold")
                    ax.grid(True, alpha=0.3, linestyle="--")
                    ax.set_xticks(year_df[fiscal_year_col])
                    plt.tight_layout()
                    figures.append((fig, f"Graph 3: Fiscal Year vs {col_title}"))
                    logger.info(f"Created Graph 3: {col_title}")
            except Exception as e:
                logger.warning(f"Could not create continuation chart: {str(e)}")

            # Visualization 4: Fiscal Year vs Change Employer (or next available column)
            try:
                # Try to find Change Employer first
                change_emp_col = None
                for col in numeric_cols:
                    if "CHANGE" in col.upper() and "EMPLOYER" in col.upper():
                        change_emp_col = col
                        break
                
                # If not found, use the next available column that hasn't been used
                if not change_emp_col:
                    used_cols = [fig[1].split("vs")[-1].strip() for fig in figures if "vs" in fig[1]]
                    for col in numeric_cols:
                        col_title = col.replace("_", " ").title()
                        if col_title not in used_cols and col.upper() != "TOTAL_APPROVALS":
                            change_emp_col = col
                            break
                
                if change_emp_col:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(data=year_df, x=fiscal_year_col, y=change_emp_col, ax=ax, 
                              palette="Set2", hue=fiscal_year_col, legend=False)
                    col_title = change_emp_col.replace("_", " ").title()
                    ax.set_title(f"Fiscal Year vs {col_title}", fontsize=16, fontweight="bold", pad=20)
                    ax.set_xlabel("Fiscal Year", fontsize=13, fontweight="bold")
                    ax.set_ylabel("Number of Approvals", fontsize=13, fontweight="bold")
                    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
                    plt.tight_layout()
                    figures.append((fig, f"Graph 4: Fiscal Year vs {col_title}"))
                    logger.info(f"Created Graph 4: {col_title}")
            except Exception as e:
                logger.warning(f"Could not create change employer chart: {str(e)}")

        # Visualization for non-time-series data: Bar chart of top values
        elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                top_n = min(15, len(df))
                top_df = df.nlargest(top_n, numeric_cols[0])
                sns.barplot(
                    data=top_df, x=numeric_cols[0], y=categorical_cols[0], ax=ax, hue=categorical_cols[0], palette="viridis", legend=False
                )
                ax.set_title(f"Top {top_n} by {numeric_cols[0]}", fontsize=14, fontweight="bold")
                ax.set_xlabel(numeric_cols[0], fontsize=12)
                ax.set_ylabel(categorical_cols[0], fontsize=12)
                plt.tight_layout()
                figures.append((fig, f"Top Values: {numeric_cols[0]} by {categorical_cols[0]}"))
            except Exception as e:
                logger.warning(f"Could not create bar chart: {str(e)}")

        # If no visualizations created yet, create a simple one using seaborn
        # IMPORTANT: Never use FISCAL_YEAR as y-axis, always use other numeric columns
        if len(figures) == 0:
            logger.warning(f"No visualizations created. has_fiscal_year: {has_fiscal_year}, fiscal_year_col: {fiscal_year_col}, numeric_cols: {numeric_cols}, categorical_cols: {categorical_cols}")
            if len(numeric_cols) > 0:
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    if has_fiscal_year and fiscal_year_col:
                        # Use FISCAL_YEAR as x-axis, other numeric columns as y-axis
                        year_df = df.sort_values(fiscal_year_col)
                        # Use the first non-FISCAL_YEAR numeric column
                        y_col = numeric_cols[0]
                        sns.lineplot(data=year_df, x=fiscal_year_col, y=y_col, marker="o", linewidth=2.5, markersize=8, ax=ax)
                        ax.set_xlabel("Fiscal Year", fontsize=12, fontweight="bold")
                        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
                        ax.set_title(f"Fiscal Year vs {y_col.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
                    else:
                        plot_df = df.copy()
                        plot_df['Index'] = range(len(plot_df))
                        sns.barplot(data=plot_df, x='Index', y=numeric_cols[0], ax=ax, palette="viridis", hue='Index', legend=False)
                        ax.set_xlabel("Index", fontsize=12, fontweight="bold")
                        ax.set_ylabel(numeric_cols[0].replace("_", " ").title(), fontsize=12, fontweight="bold")
                        ax.set_title(f"{numeric_cols[0].replace('_', ' ').title()} Distribution", fontsize=14, fontweight="bold")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    figures.append((fig, f"Basic Chart: {numeric_cols[0].replace('_', ' ').title()}"))
                except Exception as e:
                    logger.error(f"Could not create basic chart: {str(e)}", exc_info=True)

        return figures

    def process_question(self, question_key: str) -> Dict[str, Any]:
        """Process a question: execute query, generate visualizations, and summary"""
        questions = self.QUESTIONS
        if question_key not in questions:
            raise ValueError(f"Question '{question_key}' not found")

        question_data = questions[question_key]
        query = question_data["query"]
        description = question_data["description"]

        # Execute query
        df = self.execute_query(query)

        # Generate summary
        summary = self.generate_summary(question_key, df)

        # Create visualizations
        figures = self.create_visualizations(df, question_key)

        return {
            "question": question_key,
            "description": description,
            "data": df,
            "summary": summary,
            "visualizations": figures,
            "query": query,
        }

    def close(self):
        """Close SQLAlchemy engine"""
        if hasattr(self, "engine"):
            self.engine.dispose()
