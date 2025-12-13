from typing import Dict, Any, List
import json
import streamlit as st
from .llm_handler import LLMHandler
from config import Config


class JobMatcher:
    """LLM-based job matching and analysis using DeepSeek"""
    
    def __init__(self):
        # Always use DeepSeek for job search (cheap and fast)
        self.llm = LLMHandler(Config.JOB_SEARCH_MODEL)
    
    def extract_job_search_params(self, resume_text: str) -> Dict[str, Any]:
        """
        Analyze resume and extract job search parameters.
        
        Returns:
            {
                "job_titles": ["Data Scientist", "ML Engineer"],
                "locations": ["Boston, MA", "Remote"],
                "employment_types": ["FULLTIME"],
                "key_skills": ["Python", "Machine Learning", "SQL"],
                "experience_level": "mid",
                "min_salary": 80000
            }
        """
        
        prompt = f"""
You are a career advisor analyzing a resume to help find relevant jobs.

RESUME:
{resume_text[:3000]}  

Extract the following information in JSON format:

1. job_titles: List 2-3 job titles this person would be qualified for (e.g., ["Data Scientist", "ML Engineer"])
2. locations: Preferred locations mentioned or inferred (e.g., ["Boston, MA", "Remote"])
3. employment_types: Preferred types from ["FULLTIME", "PARTTIME", "CONTRACTOR", "INTERN"]
4. key_skills: Top 5-8 technical skills (e.g., ["Python", "SQL", "Machine Learning"])
5. experience_level: "entry", "mid", or "senior"
6. min_salary: Estimated minimum salary expectation (integer, or 0 if unknown)

Respond ONLY with valid JSON:

{{
  "job_titles": [],
  "locations": [],
  "employment_types": [],
  "key_skills": [],
  "experience_level": "",
  "min_salary": 0
}}
""".strip()
        
        try:
            raw_response = self.llm.generate_raw(prompt)
            return self._extract_json(raw_response)
        except Exception as e:
            st.error(f"Error extracting job search params: {str(e)}")
            return {
                "job_titles": ["Software Engineer"],
                "locations": ["Remote"],
                "employment_types": ["FULLTIME"],
                "key_skills": [],
                "experience_level": "mid",
                "min_salary": 0
            }
    
    def calculate_match_score(self, resume_text: str, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate how well a resume matches a specific job.
        
        Returns:
            {
                "match_score": 85,
                "matching_skills": ["Python", "SQL"],
                "missing_skills": ["Docker", "Kubernetes"],
                "experience_match": true,
                "education_match": true,
                "summary": "Strong match..."
            }
        """
        
        job_title = job.get("job_title", "Unknown")
        job_description = job.get("job_description", "")[:2000]
        required_skills = job.get("job_required_skills", [])
        
        # Build highlights text
        highlights = job.get("job_highlights", {})
        qualifications = "\n".join(highlights.get("Qualifications", [])[:5])
        
        prompt = f"""
You are an ATS (Applicant Tracking System) expert evaluating resume-job fit.

JOB TITLE: {job_title}

JOB DESCRIPTION (excerpt):
{job_description}

REQUIRED SKILLS: {', '.join(required_skills[:10])}

KEY QUALIFICATIONS:
{qualifications}

RESUME (excerpt):
{resume_text[:2000]}

Analyze the match and respond ONLY with valid JSON:

{{
  "match_score": 0-100,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "experience_match": true/false,
  "education_match": true/false,
  "summary": "2-3 sentence summary of fit"
}}
""".strip()
        
        try:
            raw_response = self.llm.generate_raw(prompt)
            result = self._extract_json(raw_response)
            
            # Ensure match_score is an integer
            if "match_score" in result:
                result["match_score"] = int(result["match_score"])
            else:
                result["match_score"] = 50
            
            return result
        
        except Exception as e:
            st.error(f"Error calculating match score: {str(e)}")
            return {
                "match_score": 50,
                "matching_skills": [],
                "missing_skills": [],
                "experience_match": False,
                "education_match": False,
                "summary": "Unable to calculate detailed match."
            }
    
    def generate_job_comparison(
        self,
        resume_text: str,
        job: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a detailed comparison between resume and job (like mini council).
        
        Returns:
            {
                "overall_fit": "Strong match for this role",
                "strengths": ["relevant experience", "matching skills"],
                "gaps": ["missing Docker experience"],
                "recommendations": ["Add Docker to resume", "Highlight AWS experience"]
            }
        """
        
        job_title = job.get("job_title", "Unknown")
        job_description = job.get("job_description", "")[:2500]
        
        highlights = job.get("job_highlights", {})
        qualifications = "\n".join(highlights.get("Qualifications", []))
        responsibilities = "\n".join(highlights.get("Responsibilities", []))
        
        prompt = f"""
You are a career coach helping a candidate evaluate a specific job opportunity.

JOB POSTING:
Title: {job_title}
Company: {job.get("employer_name", "Unknown")}

Description:
{job_description}

Key Qualifications:
{qualifications}

Key Responsibilities:
{responsibilities}

CANDIDATE'S RESUME:
{resume_text[:2500]}

Provide a detailed analysis in JSON format:

{{
  "overall_fit": "1-2 sentence summary",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "gaps": ["gap 1", "gap 2"],
  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"]
}}
""".strip()
        
        try:
            raw_response = self.llm.generate_raw(prompt)
            return self._extract_json(raw_response)
        
        except Exception as e:
            st.error(f"Error generating comparison: {str(e)}")
            return {
                "overall_fit": "Analysis unavailable",
                "strengths": [],
                "gaps": [],
                "recommendations": []
            }
    
    def rank_jobs_by_fit(
        self,
        resume_text: str,
        jobs: List[Dict[str, Any]],
        max_jobs: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Rank jobs by fit score (adds 'match_score' field to each job).
        Only processes first max_jobs to avoid rate limits.
        """
        
        ranked_jobs = []
        
        # Limit number of jobs to score (API rate limits)
        jobs_to_score = jobs[:max_jobs]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, job in enumerate(jobs_to_score):
            status_text.text(f"Analyzing job {i+1}/{len(jobs_to_score)}...")
            
            match_result = self.calculate_match_score(resume_text, job)
            job["match_score"] = match_result["match_score"]
            job["match_details"] = match_result
            
            ranked_jobs.append(job)
            progress_bar.progress((i + 1) / len(jobs_to_score))
        
        progress_bar.empty()
        status_text.empty()
        
        # Sort by match score (descending)
        ranked_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        
        return ranked_jobs
    
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON from potentially noisy LLM response"""
        try:
            text = text.strip()
            
            # Fast path - already clean JSON
            if text.startswith("{") and text.endswith("}"):
                return json.loads(text)
            
            # Find first { and last }
            start = text.find("{")
            end = text.rfind("}")
            
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
            
        except Exception as e:
            st.warning(f"JSON parse error: {str(e)}")
        
        # Return safe default
        return {}