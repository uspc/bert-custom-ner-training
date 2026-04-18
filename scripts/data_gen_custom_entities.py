import random
import json
from datetime import datetime, timedelta

incident_types = ["outage", "latency issue", "failure", "degradation", "crash"]
severities = ["Critical", "High", "Medium", "Low"]
applications = ["SAP", "Salesforce", "Workday", "ServiceNow", "Oracle CRM"]
systems = ["API gateway", "database", "authentication service", "payment engine"]
environments = ["prod", "staging", "dev"]
regions = ["US-East", "US-West", "EU-West", "APAC"]
error_codes = ["HTTP_500", "HTTP_503", "ORA-12154", "AUTH_FAIL"]
impacts = ["login failures", "payment delays", "data loss", "request timeout"]
root_causes = ["memory leak", "misconfiguration", "network congestion", "db lock"]
teams = ["SRE team", "DB ops", "Network team", "Platform engineering"]

def random_date():
    start = datetime(2026, 1, 1)
    return (start + timedelta(days=random.randint(0, 120))).strftime("%Y-%m-%d")

def random_time():
    return f"{random.randint(0,23):02d}:{random.randint(0,59):02d}"

def ticket():
    return f"INC{random.randint(100000,999999)}"

def find_span(text, value):
    start = text.find(value)
    return [start, start + len(value)]

def generate_text():
    sev = random.choice(severities)
    inc = random.choice(incident_types)
    app = random.choice(applications)
    sys = random.choice(systems)
    env = random.choice(environments)
    reg = random.choice(regions)
    err = random.choice(error_codes)
    impact = random.choice(impacts)
    cause = random.choice(root_causes)
    team = random.choice(teams)
    t_id = ticket()
    date = random_date()
    time = random_time()

    text = (
        f"{sev} {inc} in {app} {sys} with error {err} "
        f"causing {impact} in {env} environment for {reg} at {time} on {date}. "
        f"Ticket {t_id} assigned to {team} due to {cause}."
    )

    entities = []
    fields = [
        (sev, "SEVERITY"),
        (inc, "INCIDENT_TYPE"),
        (app, "APPLICATION"),
        (sys, "SYSTEM_COMPONENT"),
        (err, "ERROR_CODE"),
        (impact, "IMPACT"),
        (env, "ENVIRONMENT"),
        (reg, "REGION"),
        (time, "TIME"),
        (date, "DATE"),
        (t_id, "TICKET_ID"),
        (team, "TEAM"),
        (cause, "ROOT_CAUSE"),
    ]

    for value, label in fields:
        entities.append(find_span(text, value) + [label])

    return {"text": text, "entities": entities}

def generate_dataset(n=3000):
    with open("custom_incidents.jsonl", "w") as f:
        for _ in range(n):
            f.write(json.dumps(generate_text()) + "\n")

if __name__ == "__main__":
    generate_dataset(3000)
