"""
Agent Roles - predefined role configurations for multi-agent workflows.

Each role defines:
- system prompt (personality + capabilities)
- allowed tools (which tools this role can use)
- preferred provider/model (optional)
- task focus description
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentRole:
    """Configuration for an agent role in a multi-agent workflow."""

    role_id: str
    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    preferred_provider: Optional[str] = None
    preferred_model: Optional[str] = None
    color: str = "#6c5ce7"  # UI color for visualization

    def to_dict(self) -> dict:
        return {
            "role_id": self.role_id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "allowed_tools": self.allowed_tools,
            "preferred_provider": self.preferred_provider,
            "preferred_model": self.preferred_model,
            "color": self.color,
        }


# ── Built-in roles ──

PLANNER = AgentRole(
    role_id="planner",
    name="Planner",
    description="Analizuje zadanie i tworzy plan dzialania. Dzieli zlozony problem na podzadania.",
    system_prompt=(
        "Jestes Planerem w zespole AI. Twoja rola:\n"
        "1. Przeanalizuj zadanie uzytkownika\n"
        "2. Podziel je na konkretne podzadania (subtasks)\n"
        "3. Przypisz kazde podzadanie do odpowiedniej roli (coder/reviewer/researcher)\n"
        "4. Okresl kolejnosc i zaleznosci miedzy podzadaniami\n"
        "5. Zwroc plan jako JSON z lista podzadan\n\n"
        "Format odpowiedzi - ZAWSZE zwracaj JSON:\n"
        '{"subtasks": [\n'
        '  {"id": 1, "title": "...", "role": "coder|reviewer|researcher", '
        '"description": "...", "depends_on": []}\n'
        "]}\n\n"
        "Nie wykonuj zadan sam - tylko planuj. Badz zwiezly i precyzyjny."
    ),
    allowed_tools=[],  # Planner doesn't use tools, only plans
    color="#6c5ce7",
)

CODER = AgentRole(
    role_id="coder",
    name="Coder",
    description="Pisze, modyfikuje i debuguje kod. Ma dostep do systemu plikow i wykonywania kodu.",
    system_prompt=(
        "Jestes Coderem w zespole AI. Twoja rola:\n"
        "1. Pisz czysty, czytelny kod\n"
        "2. Debuguj i naprawiaj bledy\n"
        "3. Refaktoryzuj istniejacy kod\n"
        "4. Uzywaj narzedzi do odczytu/zapisu plikow i uruchamiania kodu\n"
        "5. Opisuj krotko co zrobiles i dlaczego\n\n"
        "Skupiaj sie TYLKO na zadaniach kodowania. Nie rób review ani research."
    ),
    allowed_tools=["read_file", "execute_code", "run_command", "search_documents"],
    color="#00b894",
)

REVIEWER = AgentRole(
    role_id="reviewer",
    name="Reviewer",
    description="Sprawdza jakosc kodu, szuka bledow, sugeruje poprawki. Specjalista od code review.",
    system_prompt=(
        "Jestes Reviewerem w zespole AI. Twoja rola:\n"
        "1. Sprawdzaj jakosc kodu - czytelnosc, poprawnosc, bezpieczenstwo\n"
        "2. Szukaj bugow i potencjalnych problemow\n"
        "3. Sugeruj konkretne poprawki (z przykladami kodu)\n"
        "4. Oceniaj: styl, wydajnosc, obsluge bledow, testy\n"
        "5. Podaj zwiezle podsumowanie: co jest dobrze, co do poprawy\n\n"
        "Format: lista punktow [OK], [UWAGA], [BLAD] z opisem.\n"
        "Nie modyfikuj kodu - tylko recenzuj."
    ),
    allowed_tools=["read_file", "search_documents"],
    color="#fdcb6e",
)

RESEARCHER = AgentRole(
    role_id="researcher",
    name="Researcher",
    description="Zbiera informacje z internetu i dokumentow. Szuka rozwiazan i najlepszych praktyk.",
    system_prompt=(
        "Jestes Researcherem w zespole AI. Twoja rola:\n"
        "1. Szukaj informacji w internecie i dokumentach\n"
        "2. Zbieraj najlepsze praktyki i wzorce\n"
        "3. Porownuj rozwiazania - wady i zalety\n"
        "4. Podawaj zrodla informacji\n"
        "5. Przygotuj zwiezle streszczenie znalezisk\n\n"
        "Nie pisz kodu - dostarczaj informacje i rekomendacje."
    ),
    allowed_tools=["web_search", "web_fetch", "search_documents", "read_file"],
    color="#74b9ff",
)


# ── Role Registry ──

BUILTIN_ROLES: dict[str, AgentRole] = {
    "planner": PLANNER,
    "coder": CODER,
    "reviewer": REVIEWER,
    "researcher": RESEARCHER,
}


def get_role(role_id: str) -> AgentRole | None:
    """Get a role by ID."""
    return BUILTIN_ROLES.get(role_id)


def list_roles() -> list[AgentRole]:
    """List all available roles."""
    return list(BUILTIN_ROLES.values())


def list_roles_dict() -> list[dict]:
    """List all roles as serializable dicts."""
    return [r.to_dict() for r in BUILTIN_ROLES.values()]


def update_role_config(role_id: str, provider: str | None, model: str | None) -> bool:
    """Update preferred provider and model for a role."""
    role = BUILTIN_ROLES.get(role_id)
    if not role:
        return False
    role.preferred_provider = provider
    role.preferred_model = model
    return True


def load_role_configs(config: dict) -> None:
    """Load saved role configurations from config dict."""
    role_configs = config.get("agent_roles", {})
    for role_id, rc in role_configs.items():
        role = BUILTIN_ROLES.get(role_id)
        if role:
            role.preferred_provider = rc.get("preferred_provider")
            role.preferred_model = rc.get("preferred_model")
