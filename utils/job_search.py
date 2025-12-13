import requests
from typing import Optional, List, Dict, Any
from config import Config
import streamlit as st


class JobSearchAPI:
    """Wrapper for JSearch API from RapidAPI"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.JSEARCH_API_KEY
        self.base_url = "https://jsearch.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }
    
    def search_jobs(
        self,
        query: str,
        location: Optional[str] = None,
        num_pages: int = 1,
        employment_types: Optional[List[str]] = None,
        remote_jobs_only: bool = False,
        date_posted: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for jobs using JSearch API.
        
        Args:
            query: Job title, keywords, or company name
            location: Location (e.g., "Boston, MA", "New York", "Remote")
            num_pages: Number of pages to retrieve (1-20)
            employment_types: List of types - ["FULLTIME", "CONTRACTOR", "PARTTIME", "INTERN"]
            remote_jobs_only: If True, only return remote jobs
            date_posted: "all", "today", "3days", "week", "month"
        
        Returns:
            Dictionary containing job results
        """
        
        url = f"{self.base_url}/search"
        
        # Build query parameters
        params = {
            "query": query if not location else f"{query} in {location}",
            "num_pages": str(num_pages)
        }
        
        if remote_jobs_only:
            params["remote_jobs_only"] = "true"
        
        if employment_types:
            params["employment_types"] = ",".join(employment_types)
        
        if date_posted:
            params["date_posted"] = date_posted
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching jobs: {str(e)}")
            return {"error": str(e), "data": []}
    
    def get_job_details(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific job posting.
        
        Args:
            job_id: The job ID from search results
        
        Returns:
            Dictionary containing detailed job information
        """
        
        url = f"{self.base_url}/job-details"
        params = {"job_id": job_id}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching job details: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def format_salary(job: Dict[str, Any]) -> str:
        """Format salary information from job data"""
        min_sal = job.get("job_min_salary")
        max_sal = job.get("job_max_salary")
        currency = job.get("job_salary_currency", "USD")
        period = job.get("job_salary_period", "YEAR")
        
        if not min_sal and not max_sal:
            return "Salary not disclosed"
        
        # Format with commas
        if min_sal and max_sal:
            salary_str = f"${min_sal:,} - ${max_sal:,}"
        elif min_sal:
            salary_str = f"${min_sal:,}+"
        else:
            salary_str = f"Up to ${max_sal:,}"
        
        # Add period
        period_map = {
            "YEAR": "/year",
            "MONTH": "/month",
            "HOUR": "/hour",
            "WEEK": "/week"
        }
        salary_str += period_map.get(period, "")
        
        return salary_str
    
    @staticmethod
    def format_posted_date(datetime_str: str) -> str:
        """Convert ISO datetime to relative time (e.g., '2 days ago')"""
        from datetime import datetime, timezone
        
        try:
            posted_date = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            delta = now - posted_date
            
            if delta.days == 0:
                hours = delta.seconds // 3600
                if hours == 0:
                    minutes = delta.seconds // 60
                    return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif delta.days == 1:
                return "1 day ago"
            elif delta.days < 7:
                return f"{delta.days} days ago"
            elif delta.days < 30:
                weeks = delta.days // 7
                return f"{weeks} week{'s' if weeks != 1 else ''} ago"
            elif delta.days < 365:
                months = delta.days // 30
                return f"{months} month{'s' if months != 1 else ''} ago"
            else:
                years = delta.days // 365
                return f"{years} year{'s' if years != 1 else ''} ago"
        except:
            return "Recently"
    
    @staticmethod
    def filter_jobs_by_salary(jobs: List[Dict], min_salary: int) -> List[Dict]:
        """Filter jobs by minimum salary requirement"""
        return [
            job for job in jobs
            if job.get("job_min_salary") and job["job_min_salary"] >= min_salary
        ]
    
    @staticmethod
    def filter_jobs_by_keywords(jobs: List[Dict], keywords: List[str]) -> List[Dict]:
        """Filter jobs that contain specific keywords in description"""
        filtered = []
        for job in jobs:
            description = job.get("job_description", "").lower()
            if any(keyword.lower() in description for keyword in keywords):
                filtered.append(job)
        return filtered