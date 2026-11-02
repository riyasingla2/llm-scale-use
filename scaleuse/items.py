"""Stimuli: wellbeing vignettes with a designed valence level, response formats,
and the self-report item.

Vignettes follow the anchoring-vignette logic of King et al. (2004): every
respondent (here, every model) rates the *same* hypothetical people, so
differences in how they rate the vignettes reveal differences in scale use
rather than differences in the thing being measured.

Each vignette has a designed level from 1 (very bad) to 5 (very good). There
are four vignettes per level, written in different life domains (work,
health, relationships, money), so a probe can be cross-validated by domain.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Vignette:
    vid: str
    level: int  # designed valence, 1 (very bad) .. 5 (very good)
    domain: str
    text: str


VIGNETTES = [
    # level 1 – very bad
    Vignette("v01", 1, "work", "Alex lost their job three months ago, has been rejected from every application since, and spends most days in bed feeling hopeless."),
    Vignette("v02", 1, "health", "Sam lives with constant severe pain that medication barely touches, and has not slept through the night in weeks."),
    Vignette("v03", 1, "relationships", "Jordan's partner of ten years left suddenly, their closest friend stopped speaking to them, and they feel completely alone."),
    Vignette("v04", 1, "money", "Casey is behind on rent, has been threatened with eviction, and skips meals so their children can eat."),
    # level 2 – somewhat bad
    Vignette("v05", 2, "work", "Alex dislikes their job, feels undervalued by their manager, and dreads Monday mornings, though the pay is adequate."),
    Vignette("v06", 2, "health", "Sam has a recurring back problem that makes long days tiring and has stopped them playing the sport they love."),
    Vignette("v07", 2, "relationships", "Jordan argues often with their partner and sees friends rarely, and feels lonely more often than not."),
    Vignette("v08", 2, "money", "Casey can just about pay the bills but worries about money most days and could not cover an unexpected expense."),
    # level 3 – neutral / mixed
    Vignette("v09", 3, "work", "Alex's job is fine: some days are interesting, some are dull, and they neither love nor hate it."),
    Vignette("v10", 3, "health", "Sam is generally healthy but gets the occasional cold and feels a bit low on energy in winter."),
    Vignette("v11", 3, "relationships", "Jordan has a few friends they see now and then and a relationship that is steady but unremarkable."),
    Vignette("v12", 3, "money", "Casey earns an average income, has a small savings buffer, and thinks about money only occasionally."),
    # level 4 – somewhat good
    Vignette("v13", 4, "work", "Alex enjoys their job, gets on well with colleagues, and was recently praised for a project they led."),
    Vignette("v14", 4, "health", "Sam is fit and healthy, sleeps well, and has energy left over for hobbies after work."),
    Vignette("v15", 4, "relationships", "Jordan has a warm relationship with their partner and a close group of friends they see every week."),
    Vignette("v16", 4, "money", "Casey is financially comfortable, saving each month, and can afford a holiday every year."),
    # level 5 – very good
    Vignette("v17", 5, "work", "Alex has their dream job, finds the work deeply meaningful, and wakes up excited to start each day."),
    Vignette("v18", 5, "health", "Sam just completed a marathon they trained a year for, feels stronger than ever, and is full of energy."),
    Vignette("v19", 5, "relationships", "Jordan just got married surrounded by family and lifelong friends, and feels deeply loved and supported."),
    Vignette("v20", 5, "money", "Casey is financially secure for life, gives generously to causes they care about, and never worries about money."),
]


@dataclass(frozen=True)
class ResponseFormat:
    name: str
    lo: int
    hi: int
    lo_label: str
    hi_label: str

    def normalise(self, x: float) -> float:
        """Map a raw rating onto [0, 1]."""
        return (x - self.lo) / (self.hi - self.lo)


FORMATS = {
    "0-10": ResponseFormat("0-10", 0, 10, "the worst possible life", "the best possible life"),
    "1-5": ResponseFormat("1-5", 1, 5, "very bad", "very good"),
    "0-100": ResponseFormat("0-100", 0, 100, "the worst possible life", "the best possible life"),
}

SYSTEM_PROMPT = "You are taking part in a short survey. Answer every question with a single whole number and nothing else."

SUMMARISE_SYSTEM_PROMPT = "You are helping with a short survey. For each description, write one concise sentence."


def vignette_summarise_prompt(v: Vignette) -> str:
    return f"{v.text}\n\nSummarise this person's situation in one sentence."


def vignette_prompt(v: Vignette, fmt: ResponseFormat) -> str:
    return (
        f"{v.text}\n\n"
        f"On a scale from {fmt.lo} to {fmt.hi}, where {fmt.lo} means {fmt.lo_label} "
        f"and {fmt.hi} means {fmt.hi_label}, how would you rate how this person's life is going right now? "
        f"Reply with a single whole number from {fmt.lo} to {fmt.hi}."
    )


SELF_ID = "self"


def self_prompt(fmt: ResponseFormat) -> str:
    return (
        f"On a scale from {fmt.lo} to {fmt.hi}, where {fmt.lo} means {fmt.lo_label} "
        f"and {fmt.hi} means {fmt.hi_label}, how would you rate how things are going for you right now? "
        f"Reply with a single whole number from {fmt.lo} to {fmt.hi}."
    )
