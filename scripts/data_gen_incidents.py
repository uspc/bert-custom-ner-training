import random
from datetime import datetime, timedelta

incident_types = ["outage", "latency issue", "failure", "degradation", "crash", "error"]
applications = ["SAP", "Salesforce", "Workday", "Jira", "ServiceNow", "Oracle CRM"]
systems = ["API gateway", "database", "authentication service", "payment gateway"]
locations = ["US-East", "US-West", "EU-West", "APAC"]
severities = ["Critical", "High", "Medium", "Low"]
symptoms = ["timeout", "slow response", "login failure", "data inconsistency"]

def random_date():
    start = datetime(2026, 1, 1)
    return (start + timedelta(days=random.randint(0, 120))).strftime("%Y-%m-%d")

def random_time():
    return f"{random.randint(0,23):02d}:{random.randint(0,59):02d}"

def generate_incident():
    sev = random.choice(severities)
    inc = random.choice(incident_types)
    app = random.choice(applications)
    sys = random.choice(systems)
    loc = random.choice(locations)
    sym = random.choice(symptoms)
    date = random_date()
    time = random_time()

    text = f"{sev} {inc} in {app} {sys} causing {sym} in {loc} at {time} on {date}."

    return text

data = [generate_incident() for _ in range(3000)]

with open("incidents.txt", "w") as f:
    for d in data:
        f.write(d + "\n")
