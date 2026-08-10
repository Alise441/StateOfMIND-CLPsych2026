"""
MIND ABCD Taxonomy for CLPsych 2026.

All definitions from CLPsych 2026 Guidelines (Clpsych2026_Guidelines.pdf).
Single source of truth for elements, subelements, valences, and presence scale.
"""

# ── Elements ──────────────────────────────────────────────────────────────────

ELEMENTS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]

ELEMENT_DEFINITIONS = {
    "A": "Affect (A): Type of emotion or mood expressed by the writer.",
    "B-O": "Behavior toward Others (B-O): The writer's main behavior(s) or tendencies directed toward other people.",
    "B-S": "Behavior toward Self (B-S): The writer's main behavior(s) or tendencies directed toward the self.",
    "C-O": "Cognition of the Other (C-O): The writer's main perceptions, beliefs, or interpretations about other people.",
    "C-S": "Cognition of the Self (C-S): The writer's main self-perceptions, beliefs, or appraisals about themselves.",
    "D": "Desire (D): The writer's main desire, expectation, need, intention, wish, or fear.",
}

# ── Valence definitions ───────────────────────────────────────────────────────

VALENCE_DEFINITIONS = {
    "adaptive": "An adaptive self-state pertains to aspects of Affect, Behavior, Cognition, and Desire that are conducive to the fulfillment of basic desires and needs.",
    "maladaptive": "A maladaptive self-state pertains to aspects of Affect, Behavior, Cognition, and Desire that hinder the fulfillment of basic desires and needs.",
}

# ── Subelements ───────────────────────────────────────────────────────────────

SUBELEMENTS = {
    "A": {
        "adaptive": {
            1: "Calm/laid back — feeling at ease, relaxed, or peacefully accepting of one's situation",
            3: "Sad/emotional pain/grieving — expressing sorrow or grief in a healthy, processing way rather than being stuck in despair",
            5: "Content/happy/joy/hopeful — feeling satisfied, cheerful, or optimistic about the present or future",
            7: "Vigor/energetic — feeling motivated, lively, or full of energy to engage with life",
            9: "Justifiable/assertive anger — expressing appropriate anger or standing up for oneself in a constructive way",
            11: "Proud — feeling a sense of achievement, self-worth, or pride in one's actions or identity",
            13: "Feel loved/belong — feeling connected, valued, or accepted by others",
        },
        "maladaptive": {
            2: "Anxious/fearful/tense — feeling worried, nervous, scared, or on edge about something",
            4: "Depressed/despair/hopeless — feeling deeply sad, empty, worthless, or seeing no way forward",
            6: "Mania — showing elevated mood, grandiosity, impulsivity, or racing thoughts beyond normal excitement",
            8: "Apathetic/blunted — feeling emotionally flat, numb, or unable to care about anything",
            10: "Angry/aggression/disgust/contempt — feeling hostile, bitter, resentful, or disgusted in a destructive way",
            12: "Ashamed/guilty — feeling intense shame, self-blame, or guilt that is disproportionate or harmful",
            14: "Feel lonely — feeling isolated, disconnected, or painfully alone even when others may be present",
        },
    },
    "B-O": {
        "adaptive": {
            1: "Relating behavior — actively reaching out, connecting, or cooperating with others in a healthy way",
            3: "Autonomous or adaptive control behavior — taking independent action or setting healthy boundaries with others",
        },
        "maladaptive": {
            2: "Fight or flight behavior — reacting to others with aggression, withdrawal, or avoidance driven by fear or threat",
            4: "Over-controlled or controlling behavior — excessively managing, manipulating, or dominating interactions with others",
        },
    },
    "B-S": {
        "adaptive": {
            1: "Self-care and improvement — taking steps to look after oneself physically, mentally, or emotionally",
        },
        "maladaptive": {
            2: "Self-harm, neglect and avoidance — engaging in self-destructive behavior, ignoring one's needs, or avoiding necessary self-care",
        },
    },
    "C-O": {
        "adaptive": {
            1: "Perception of other as related — viewing others as caring, supportive, or emotionally connected",
            3: "Perception of other as facilitating autonomy — seeing others as encouraging independence or respecting one's choices",
        },
        "maladaptive": {
            2: "Perception of other as detached or over-attached — seeing others as emotionally unavailable, intrusive, or smothering",
            4: "Perception of other as blocking autonomy — viewing others as controlling, restricting, or undermining one's independence",
        },
    },
    "C-S": {
        "adaptive": {
            1: "Self-acceptance and compassion — treating oneself with kindness, understanding, or accepting one's flaws without harsh judgment",
        },
        "maladaptive": {
            2: "Self-criticism — harshly judging, blaming, or devaluing oneself in a way that undermines well-being",
        },
    },
    "D": {
        "adaptive": {
            1: "Relatedness — wanting to connect, bond, or maintain meaningful relationships with others",
            3: "Autonomy and adaptive control — desiring independence, self-direction, or healthy control over one's life",
            5: "Competence/self-esteem/self-care — wanting to feel capable, worthy, or to take good care of oneself",
        },
        "maladaptive": {
            2: "Expectation that relatedness needs will not be met — believing that one will remain unloved, rejected, or unable to form connections",
            4: "Expectation that autonomy needs will not be met — believing one will remain trapped, powerless, or unable to control one's life",
            6: "Expectation that competence needs will not be met — believing one will remain incompetent, worthless, or unable to improve",
        },
    },
}

# ── Presence scale ────────────────────────────────────────────────────────────

PRESENCE_SCALE = {
    1: "Not present: The self state is not expressed in the post.",
    2: "Somewhat present: The self state is expressed, but plays a subtle, limited role.",
    3: "Moderately present: The self state is clearly expressed and moderately contributes to the person's experience.",
    4: "Much present: The self state strongly influences and shapes the experience described in the post.",
    5: "Highly present: The self state strongly shapes and clearly defines the overall experience described in the post.",
}

