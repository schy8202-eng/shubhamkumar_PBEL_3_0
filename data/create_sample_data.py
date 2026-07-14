import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

REAL_TEMPLATES = [
    "The {org} announced on {day} that it will {action} in response to {topic}, according to officials familiar with the matter.",
    "Researchers at {org} published a study in a peer-reviewed journal showing {topic} may {action}, the report said.",
    "{org} reported quarterly earnings that {action}, meeting analyst expectations for {topic}.",
    "Local officials confirmed that {org} has begun work to {action} as part of an ongoing initiative on {topic}.",
    "According to a statement released by {org}, the agency plans to {action} following months of review on {topic}.",
    "Government data released this week shows {topic} has {action}, prompting a policy response from {org}.",
    "{org} officials held a press conference to clarify recent developments regarding {topic} and outline plans to {action}.",
    "A new report from {org} details how {topic} could {action} over the next fiscal year.",
]

FAKE_TEMPLATES = [
    "SHOCKING: You won't believe what {org} is hiding about {topic} — {action}, insiders reveal!!!",
    "BREAKING!!! Secret documents PROVE {org} covered up {topic}, and they {action}, sources CLAIM.",
    "Doctors HATE this: {topic} can {action}, and {org} doesn't want you to know!!!",
    "EXPOSED: {org} secretly plans to {action} using {topic} while media stays SILENT.",
    "Miracle discovery: {topic} will {action}, but {org} is trying to BAN it from the public!!!",
    "You WON'T believe this insane conspiracy involving {org} and {topic} — they {action}!!!",
    "ALERT: {org} caught RED-HANDED manipulating {topic} to {action}, anonymous whistleblower says.",
    "This ONE WEIRD TRICK about {topic} has {org} FURIOUS because it will {action}!!!",
]

ORGS = ["the government", "a major tech company", "the health ministry", "a leading university",
        "the central bank", "a pharmaceutical giant", "the city council", "a global corporation",
        "the education department", "a research institute"]
TOPICS = ["climate policy", "the economy", "public health", "national security", "vaccine rollout",
          "election integrity", "artificial intelligence regulation", "energy prices",
          "immigration reform", "social media privacy"]
ACTIONS = ["change how funding is allocated", "trigger a major shift in public opinion",
           "affect millions of citizens", "reshape industry standards", "cause widespread controversy",
           "lead to new regulations", "spark nationwide debate", "influence upcoming elections"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

def generate_article(templates):
    t = random.choice(templates)
    return t.format(
        org=random.choice(ORGS).capitalize(),
        topic=random.choice(TOPICS),
        action=random.choice(ACTIONS),
        day=random.choice(DAYS),
    )

def build_dataset(n_per_class=500):
    rows = []
    for _ in range(n_per_class):
        real_text = " ".join(generate_article(REAL_TEMPLATES) for _ in range(random.randint(2, 4)))
        fake_text = " ".join(generate_article(FAKE_TEMPLATES) for _ in range(random.randint(2, 4)))
        rows.append({"text": real_text, "label": "REAL"})
        rows.append({"text": fake_text, "label": "FAKE"})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = build_dataset(n_per_class=500)
    out_path = os.path.join(os.path.dirname(__file__), "news_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset created: {out_path}")
    print(f"Shape: {df.shape}")
    print(df['label'].value_counts())
