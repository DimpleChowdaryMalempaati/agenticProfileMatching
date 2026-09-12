"""Generate 30+ diverse synthetic resumes and 5+ job descriptions."""

from __future__ import annotations

import json
from pathlib import Path

from config import JOB_DESCRIPTIONS_DIR, RESUMES_DIR

FIRST_NAMES = [
    "Alice", "Bob", "Carlos", "Diana", "Ethan", "Fatima", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Lena", "Marcus", "Nina", "Omar", "Priya",
    "Quinn", "Rachel", "Samuel", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yuki", "Zara", "Adam", "Bella", "Chris", "Deepa", "Elena", "Frank",
    "Grace", "Henry",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Lee", "Wilson", "Anderson", "Taylor",
    "Thomas", "Moore", "Jackson", "Martin", "Thompson", "White", "Harris",
    "Clark", "Lewis", "Walker", "Hall", "Allen", "Young", "King", "Wright",
    "Scott", "Green", "Baker", "Adams", "Nelson",
]

ROLE_PROFILES = [
    {
        "title": "Senior Python ML Engineer",
        "years": 8,
        "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "AWS", "Docker", "SQL"],
        "focus": "Built production ML pipelines and recommendation systems.",
    },
    {
        "title": "Full Stack Developer",
        "years": 5,
        "skills": ["JavaScript", "TypeScript", "React", "Node.js", "PostgreSQL", "REST", "Git"],
        "focus": "Developed scalable web applications and REST APIs.",
    },
    {
        "title": "Data Scientist",
        "years": 6,
        "skills": ["Python", "Pandas", "Scikit-learn", "SQL", "Data Analysis", "Machine Learning"],
        "focus": "Delivered predictive models and business analytics dashboards.",
    },
    {
        "title": "DevOps Engineer",
        "years": 7,
        "skills": ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD", "Linux", "Python"],
        "focus": "Automated cloud infrastructure and deployment pipelines.",
    },
    {
        "title": "Backend Java Developer",
        "years": 9,
        "skills": ["Java", "Spring", "Microservices", "PostgreSQL", "Kafka", "REST"],
        "focus": "Designed microservices for high-traffic financial platforms.",
    },
    {
        "title": "NLP Engineer",
        "years": 4,
        "skills": ["Python", "NLP", "PyTorch", "Machine Learning", "FastAPI", "Docker"],
        "focus": "Built chatbots and text classification systems.",
    },
    {
        "title": "Cloud Architect",
        "years": 12,
        "skills": ["AWS", "Azure", "GCP", "Terraform", "Kubernetes", "DevOps"],
        "focus": "Led cloud migration and architecture for enterprise clients.",
    },
    {
        "title": "Frontend Engineer",
        "years": 4,
        "skills": ["React", "TypeScript", "JavaScript", "GraphQL", "REST", "Git"],
        "focus": "Created responsive UI components and design systems.",
    },
    {
        "title": "Data Engineer",
        "years": 6,
        "skills": ["Python", "Spark", "SQL", "AWS", "Airflow", "PostgreSQL"],
        "focus": "Built ETL pipelines and data warehouses.",
    },
    {
        "title": "Mobile Developer",
        "years": 5,
        "skills": ["JavaScript", "React", "TypeScript", "REST", "Git", "Agile"],
        "focus": "Shipped cross-platform mobile apps with React Native.",
    },
]

UNIVERSITIES = [
    "MIT", "Stanford University", "UC Berkeley", "Carnegie Mellon University",
    "Georgia Tech", "University of Michigan", "UT Austin", "Purdue University",
    "University of Washington", "Cornell University",
]

DEGREES = [
    "B.S. Computer Science", "M.S. Computer Science", "B.S. Data Science",
    "M.S. Data Science", "B.S. Software Engineering", "M.S. Machine Learning",
    "B.A. Mathematics", "Ph.D. Computer Science",
]


def _resume_template(name: str, profile: dict, index: int) -> str:
    uni = UNIVERSITIES[index % len(UNIVERSITIES)]
    degree = DEGREES[index % len(DEGREES)]
    skills_line = ", ".join(profile["skills"])
    start_year = 2026 - int(profile["years"]) - 2

    return f"""{name}
{name.lower().replace(' ', '.')}@email.com | linkedin.com/in/{name.lower().replace(' ', '')}

Summary
{profile['title']} with {profile['years']}+ years of experience. {profile['focus']}

Experience
Senior {profile['title']} | TechCorp {index + 1}
{start_year + 2}-Present
- {profile['focus']}
- Led cross-functional teams and improved system reliability by 35%.
- Technologies: {skills_line}

{profile['title']} | InnovateSoft {index + 1}
{start_year}-{start_year + 2}
- Developed core features and automated testing workflows.
- Collaborated in Agile/Scrum environment.

Education
{degree}, {uni}
Graduated {start_year}

Skills
{skills_line}

Projects
- Open-source contributor and internal tooling for deployment automation.
- Built analytics dashboard used by 500+ stakeholders.
"""


JOB_DESCRIPTIONS = [
    {
        "filename": "senior_python_ml_engineer.json",
        "title": "Senior Python ML Engineer",
        "description": (
            "We are seeking a Senior Python ML Engineer to design and deploy "
            "machine learning models at scale. You will work with TensorFlow/PyTorch, "
            "build data pipelines, and collaborate with product teams. "
            "Must have 5+ years Python experience and strong ML fundamentals."
        ),
        "critical_skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "AWS", "Docker"],
        "must_have": [
            {"skill": "Python", "min_years": 5, "description": "5+ years Python"},
            {"skill": "Machine Learning", "min_years": 3, "description": "ML experience"},
        ],
    },
    {
        "filename": "full_stack_developer.json",
        "title": "Full Stack Developer",
        "description": (
            "Join our product team as a Full Stack Developer. Build React frontends "
            "and Node.js backends, design REST APIs, and work with PostgreSQL. "
            "3+ years experience required. Remote-friendly."
        ),
        "critical_skills": ["React", "Node.js", "TypeScript", "PostgreSQL", "REST", "JavaScript"],
        "must_have": [
            {"skill": "React", "min_years": 2},
            {"skill": "JavaScript", "min_years": 3},
        ],
    },
    {
        "filename": "devops_engineer.json",
        "title": "DevOps Engineer",
        "description": (
            "DevOps Engineer needed to manage AWS infrastructure, Kubernetes clusters, "
            "and CI/CD pipelines. Experience with Terraform and Docker required. "
            "5+ years in DevOps or SRE roles."
        ),
        "critical_skills": ["AWS", "Kubernetes", "Docker", "Terraform", "CI/CD", "Linux"],
        "must_have": [
            {"skill": "AWS", "min_years": 4},
            {"skill": "Kubernetes", "min_years": 2},
        ],
    },
    {
        "filename": "data_scientist.json",
        "title": "Data Scientist",
        "description": (
            "Data Scientist to build predictive models and deliver insights. "
            "Strong Python, Pandas, Scikit-learn, and SQL skills. "
            "Experience with A/B testing and statistical modeling preferred."
        ),
        "critical_skills": ["Python", "Pandas", "Scikit-learn", "SQL", "Data Analysis", "Machine Learning"],
        "must_have": [
            {"skill": "Python", "min_years": 3},
            {"skill": "SQL", "min_years": 2},
        ],
    },
    {
        "filename": "nlp_engineer.json",
        "title": "NLP Engineer",
        "description": (
            "NLP Engineer to develop text classification, entity extraction, and "
            "LLM-powered features. PyTorch and FastAPI experience required. "
            "4+ years in NLP or ML engineering."
        ),
        "critical_skills": ["Python", "NLP", "PyTorch", "Machine Learning", "FastAPI"],
        "must_have": [
            {"skill": "NLP", "min_years": 3},
            {"skill": "Python", "min_years": 4},
        ],
    },
    {
        "filename": "backend_java_developer.json",
        "title": "Backend Java Developer",
        "description": (
            "Backend Java Developer for microservices platform. Spring Boot, Kafka, "
            "PostgreSQL, and REST API design. 5+ years Java experience."
        ),
        "critical_skills": ["Java", "Spring", "Microservices", "PostgreSQL", "Kafka", "REST"],
        "must_have": [
            {"skill": "Java", "min_years": 5},
            {"skill": "Spring", "min_years": 3},
        ],
    },
]


def generate_resumes(count: int = 34) -> list[Path]:
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for i in range(count):
        first = FIRST_NAMES[i % len(FIRST_NAMES)]
        last = LAST_NAMES[i % len(LAST_NAMES)]
        name = f"{first} {last}"
        profile = ROLE_PROFILES[i % len(ROLE_PROFILES)]
        filename = f"{first.lower()}_{last.lower()}.txt"
        content = _resume_template(name, profile, i)
        path = RESUMES_DIR / filename
        path.write_text(content, encoding="utf-8")
        paths.append(path)

    return paths


def generate_job_descriptions() -> list[Path]:
    JOB_DESCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for jd in JOB_DESCRIPTIONS:
        path = JOB_DESCRIPTIONS_DIR / jd["filename"]
        path.write_text(json.dumps(jd, indent=2), encoding="utf-8")
        paths.append(path)

    return paths


def main() -> None:
    resume_paths = generate_resumes(34)
    jd_paths = generate_job_descriptions()
    print(f"Generated {len(resume_paths)} resumes in {RESUMES_DIR}")
    print(f"Generated {len(jd_paths)} job descriptions in {JOB_DESCRIPTIONS_DIR}")


if __name__ == "__main__":
    main()
