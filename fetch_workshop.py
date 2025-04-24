import random

# 🔹 Predefined 50 Workshop Names
WORKSHOPS = [
    "AI & Machine Learning Bootcamp", "Cloud Computing Essentials", "Cybersecurity Fundamentals",
    "Data Science with Python", "Web Development Masterclass", "Full-Stack Development",
    "Blockchain for Beginners", "Game Development with Unity", "AR & VR Workshop",
    "Internet of Things (IoT) Basics", "Digital Marketing Crash Course", "SEO & Content Marketing",
    "Graphic Design with Adobe Suite", "UI/UX Design Fundamentals", "Mobile App Development",
    "Python for Beginners", "Java Programming Workshop", "C++ Data Structures and Algorithms",
    "Embedded Systems Workshop", "Networking & Ethical Hacking", "Soft Skills & Leadership",
    "Public Speaking & Communication", "Entrepreneurship & Startups", "Resume Building & Job Search",
    "Freelancing & Remote Work", "Finance & Investment Basics", "Stock Market & Trading Workshop",
    "Personal Branding & Social Media", "Creative Writing & Blogging", "Photography & Video Editing",
    "Podcasting & Audio Production", "3D Printing & Prototyping", "Mechanical Engineering Workshop",
    "Electrical Circuit Design", "Robotics & Automation", "Quantum Computing Introduction",
    "Cloud DevOps with AWS & Azure", "Google Cloud Certification Prep", "Kubernetes & Docker Hands-On",
    "Linux Administration Basics", "Cyber Forensics & Incident Response", "DevOps CI/CD Pipelines",
    "Penetration Testing & Bug Bounty", "Big Data & Analytics with Hadoop", "TensorFlow & Deep Learning",
    "ReactJS & Modern Web Apps", "Node.js & Backend Development", "Database Management with SQL",
    "Business Analytics & Excel", "Excel for Data Analysis"
]

def get_workshop():
    """Return a random workshop name from the predefined list."""
    return random.choice(WORKSHOPS)  # Select a random workshop
