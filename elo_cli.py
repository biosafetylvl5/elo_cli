#!/usr/bin/env python3
import argparse
import logging
import os
import random
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich_argparse import RichHelpFormatter

# Install Rich traceback handler for prettier error output
install_rich_traceback()

# Configure root logger to use RichHandler
logging.basicConfig(
    level="WARNING",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Shared Rich console for pretty printing
console = Console()

DB_PATH = os.getenv("ELO_DB_PATH", "elo_system.db")
ENTITIES_TABLE = "entities"
MATCHES_TABLE = "matches"


def load_yaml(path: str) -> Any:
    """
    Load and return YAML content from the given file path.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    Any
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If the file cannot be parsed.
    """
    logger.debug(f"Loading YAML from path: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    logger.debug(f"YAML loaded: {data!r}")
    return data


def save_yaml(data: Any, path: str) -> None:
    """
    Save Python object as YAML to the given path.

    Parameters
    ----------
    data : Any
        Data to serialize.
    path : str
        Path to save YAML file.

    Raises
    ------
    IOError
        If the file cannot be written.
    """
    logger.debug(f"Saving YAML data to path: {path}")
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    logger.info(f"YAML data saved to [bold]{path}[/]")


class DataStore:
    """Handles all database interactions: schema, CRUD for entities and matches."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        """
        Initialize connection and ensure schema exists.

        Parameters
        ----------
        db_path : str
            Path to SQLite database file.
        """
        logger.debug(f"Connecting to database at {db_path}")
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they do not exist."""
        logger.debug("Initializing database schema")
        cur = self.conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {ENTITIES_TABLE} (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                rating REAL NOT NULL
            )
        """.strip()
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MATCHES_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_a TEXT NOT NULL,
                entity_b TEXT NOT NULL,
                score_a REAL NOT NULL,
                score_b REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(entity_a) REFERENCES {ENTITIES_TABLE}(id),
                FOREIGN KEY(entity_b) REFERENCES {ENTITIES_TABLE}(id)
            )
        """.strip()
        )
        self.conn.commit()
        logger.info("Database schema initialized")

    def reset(self) -> None:
        """Drop all tables and reinitialize schema."""
        logger.warning("Resetting database schema")
        cur = self.conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {MATCHES_TABLE}")
        cur.execute(f"DROP TABLE IF EXISTS {ENTITIES_TABLE}")
        self.conn.commit()
        self._init_schema()
        logger.info("Database reset complete")

    def add_entities(
        self, entities: List[Dict[str, str]], initial_rating: float
    ) -> None:
        """
        Insert or replace entities with the given initial rating.

        Parameters
        ----------
        entities : List[Dict[str, str]]
            List of entities with 'id' and optional 'name'.
        initial_rating : float
            Rating to assign to each entity.
        """
        logger.debug(
            f"Adding {len(entities)} entities with initial rating {initial_rating}"
        )
        cur = self.conn.cursor()
        for ent in entities:
            ent_id = ent["id"]
            ent_name = ent.get("name", ent_id)
            logger.debug(
                f"Upserting entity {ent_id} ({ent_name}) with rating {initial_rating}"
            )
            cur.execute(
                f"INSERT OR REPLACE INTO {ENTITIES_TABLE} (id, name, rating) VALUES (?, ?, ?)",
                (ent_id, ent_name, initial_rating),
            )
        self.conn.commit()
        logger.info("Entities added/updated")

    def get_all_entities(self) -> List[Tuple[str, str, float]]:
        """Return list of (id, name, rating)."""
        logger.debug("Fetching all entities")
        cur = self.conn.cursor()
        cur.execute(f"SELECT id, name, rating FROM {ENTITIES_TABLE}")
        rows = cur.fetchall()
        logger.debug(f"Fetched {len(rows)} entities")
        return rows

    def count_matches(self, entity_id: str) -> int:
        """
        Return number of matches involving a given entity.

        Parameters
        ----------
        entity_id : str
            Entity ID to count matches for.

        Returns
        -------
        int
            Total matches played by the entity.
        """
        logger.debug(f"Counting matches for entity {entity_id}")
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT COUNT(*) FROM {MATCHES_TABLE} WHERE entity_a=? OR entity_b=?",
            (entity_id, entity_id),
        )
        count = cur.fetchone()[0]
        logger.debug(f"Entity {entity_id} has {count} matches")
        return count

    def update_rating(self, entity_id: str, new_rating: float) -> None:
        """
        Set the rating for a specific entity.

        Parameters
        ----------
        entity_id : str
            Entity ID to update.
        new_rating : float
            New rating value.
        """
        logger.info(f"Updating rating for {entity_id} to {new_rating}")
        cur = self.conn.cursor()
        cur.execute(
            f"UPDATE {ENTITIES_TABLE} SET rating=? WHERE id=?", (new_rating, entity_id)
        )
        self.conn.commit()

    def record_match(
        self,
        entity_a: str,
        entity_b: str,
        score_a: float,
        score_b: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Insert a match record with timestamp.

        Parameters
        ----------
        entity_a : str
            ID of first entity.
        entity_b : str
            ID of second entity.
        score_a : float
            Score of first entity.
        score_b : float
            Score of second entity.
        timestamp : str, optional
            ISO timestamp; if None, current UTC time is used.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        logger.debug(
            f"Recording match: {entity_a} vs {entity_b}, scores=({score_a}, {score_b}) at {ts}"
        )
        cur = self.conn.cursor()
        cur.execute(
            f"INSERT INTO {MATCHES_TABLE} (entity_a, entity_b, score_a, score_b, timestamp)"
            " VALUES (?, ?, ?, ?, ?)",
            (entity_a, entity_b, score_a, score_b, ts),
        )
        self.conn.commit()

    def get_match_history(self) -> List[Tuple[str, str, float, float, str]]:
        """Return full match history sorted by insertion order."""
        logger.debug("Fetching match history")
        cur = self.conn.cursor()
        cur.execute(
            f"SELECT entity_a, entity_b, score_a, score_b, timestamp FROM {MATCHES_TABLE}"
            " ORDER BY id ASC"
        )
        history = cur.fetchall()
        logger.debug(f"Fetched {len(history)} matches")
        return history


class EloSystem:
    """Elo rating calculation."""

    def __init__(
        self,
        k_factor: float,
        provisional_k: Optional[float] = None,
        provisional_threshold: int = 10,
    ) -> None:
        """
        Initialize Elo system.

        Parameters
        ----------
        k_factor : float
            Standard K-factor.
        provisional_k : float, optional
            K-factor for provisional players. Defaults to 2*k_factor.
        provisional_threshold : int
            Number of matches under which provisional_k applies.
        """
        self.k_factor = k_factor
        self.provisional_k = provisional_k or k_factor * 2
        self.provisional_threshold = provisional_threshold
        logger.debug(
            f"EloSystem initialized: k_factor={self.k_factor}, "
            f"provisional_k={self.provisional_k}, threshold={self.provisional_threshold}"
        )

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """
        Compute expected score for player A against B.

        Parameters
        ----------
        rating_a : float
            Rating of player A.
        rating_b : float
            Rating of player B.

        Returns
        -------
        float
            Expected score between 0 and 1.
        """
        exp = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        logger.debug(
            f"Expected score: rating_a={rating_a}, rating_b={rating_b} -> {exp}"
        )
        return exp

    def update_ratings(
        self,
        rating_a: float,
        rating_b: float,
        score_a: float,
        score_b: float,
        matches_a: int = 0,
        matches_b: int = 0,
    ) -> Tuple[float, float]:
        """
        Update player ratings after a match.

        Parameters
        ----------
        rating_a : float
            Current rating of player A.
        rating_b : float
            Current rating of player B.
        score_a : float
            Actual score of A (1 win, 0 loss, 0.5 draw).
        score_b : float
            Actual score of B.
        matches_a : int
            Number of prior matches for A.
        matches_b : int
            Number of prior matches for B.

        Returns
        -------
        Tuple[float, float]
            New ratings (new_a, new_b).
        """
        e_a = self.expected_score(rating_a, rating_b)
        e_b = 1 - e_a
        k_a = (
            self.provisional_k
            if matches_a < self.provisional_threshold
            else self.k_factor
        )
        k_b = (
            self.provisional_k
            if matches_b < self.provisional_threshold
            else self.k_factor
        )
        new_a = rating_a + k_a * (score_a - e_a)
        new_b = rating_b + k_b * (score_b - e_b)
        logger.info(
            f"Updated ratings: A {rating_a:.1f}→{new_a:.1f}, B {rating_b:.1f}→{new_b:.1f}"
        )
        return new_a, new_b


def random_pairing(
    entities: List[Tuple[str, str, float]]
) -> Tuple[Tuple[str, str, float], Tuple[str, str, float]]:
    """Return a random pair of distinct entities."""
    pair = tuple(random.sample(entities, 2))
    logger.debug(f"Random pairing selected: {pair[0][0]} vs {pair[1][0]}")
    return pair


def smart_pairing(
    entities: List[Tuple[str, str, float]], max_diff: float = 100.0
) -> Tuple[Tuple[str, str, float], Tuple[str, str, float]]:
    """
    Pair two entities with closest ratings within max_diff.

    Parameters
    ----------
    entities : List[Tuple[str, str, float]]
        List of (id, name, rating).
    max_diff : float
        Maximum allowed rating difference.

    Returns
    -------
    Tuple[Tuple[str, str, float], Tuple[str, str, float]]
    """
    sorted_ents = sorted(entities, key=lambda x: x[2])
    for a, b in zip(sorted_ents, sorted_ents[1:]):
        if abs(a[2] - b[2]) <= max_diff:
            logger.debug(f"Smart pairing selected: {a[0]} vs {b[0]}")
            return a, b
    return random_pairing(entities)


def prompt_match_outcome(name_a: str, name_b: str) -> Tuple[float, float]:
    """
    Prompt user for match outcome.

    Parameters
    ----------
    name_a : str
        Name of first player.
    name_b : str
        Name of second player.

    Returns
    -------
    Tuple[float, float]
        Scores (score_a, score_b).

    Raises
    ------
    SystemExit
        If user selects 'quit'.
    """
    prompt = f"Enter result [1={name_a}, 2={name_b}, d=draw, quit=quit]: "
    while True:
        choice = console.input(f"[bold cyan]?[/] {prompt}").strip().lower()
        if choice == "1":
            return 1.0, 0.0
        if choice == "2":
            return 0.0, 1.0
        if choice in ("d", "t"):
            return 0.5, 0.5
        if choice == "quit":
            logger.info("User quit prompt")
            raise SystemExit
        logger.warning(f"Invalid input: {choice!r}")


def play_single_match(
    datastore: DataStore, elo: EloSystem, smart: bool = False
) -> None:
    """
    Play one match and update ratings.

    Parameters
    ----------
    datastore : DataStore
        Data store instance for DB operations.
    elo : EloSystem
        Elo rating system instance.
    smart : bool
        Whether to use smart pairing.
    """
    ents = datastore.get_all_entities()
    ent_a, ent_b = smart_pairing(ents) if smart else random_pairing(ents)
    logger.info(f"Starting single match: {ent_a[1]} vs {ent_b[1]}")
    console.rule(
        f"[green]Match[/] {ent_a[1]} ([yellow]{ent_a[2]:.1f}[/]) vs {ent_b[1]} ([yellow]{ent_b[2]:.1f}[/])"
    )
    try:
        score_a, score_b = prompt_match_outcome(ent_a[1], ent_b[1])
    except SystemExit:
        logger.info("Single match canceled by user")
        console.print("[red]Match canceled.[/]")
        return

    matches_a = datastore.count_matches(ent_a[0])
    matches_b = datastore.count_matches(ent_b[0])
    new_a, new_b = elo.update_ratings(
        ent_a[2], ent_b[2], score_a, score_b, matches_a, matches_b
    )

    datastore.update_rating(ent_a[0], new_a)
    datastore.update_rating(ent_b[0], new_b)
    datastore.record_match(ent_a[0], ent_b[0], score_a, score_b)

    console.print(
        f"[bold]Updated ratings[/]: {ent_a[1]} → {new_a:.1f}, {ent_b[1]} → {new_b:.1f}"
    )


def play_continuous(
    datastore: DataStore, elo: EloSystem, smart: bool = False, assume_win: bool = False
) -> None:
    """
    Play matches repeatedly until user quits. Optionally auto-assign wins.

    Parameters
    ----------
    datastore : DataStore
    elo : EloSystem
    smart : bool
    assume_win : bool
    """
    console.print("[blue]Entering continuous play mode. Type 'quit' to stop.[/]")
    last_pair: Optional[Tuple[str, str]] = None
    last_winner: Optional[str] = None

    while True:
        ents = datastore.get_all_entities()
        ent_a, ent_b = smart_pairing(ents) if smart else random_pairing(ents)
        pair_ids = (ent_a[0], ent_b[0])

        if assume_win and last_pair and set(pair_ids) == set(last_pair):
            if last_winner == ent_a[0]:
                score_a, score_b = 1.0, 0.0
                winner, loser = ent_a, ent_b
            else:
                score_a, score_b = 0.0, 1.0
                winner, loser = ent_b, ent_a
            console.print(f"[magenta]Auto:[/] {winner[1]} wins again over {loser[1]}")
        else:
            console.rule(f"[green]Match[/] {ent_a[1]} vs {ent_b[1]}")
            try:
                score_a, score_b = prompt_match_outcome(ent_a[1], ent_b[1])
            except SystemExit:
                logger.info("Continuous play mode exited by user")
                console.print("[red]Exiting continuous play mode.[/]")
                break
            winner = (
                ent_a if score_a > score_b else (ent_b if score_b > score_a else None)
            )

        matches_a = datastore.count_matches(ent_a[0])
        matches_b = datastore.count_matches(ent_b[0])
        new_a, new_b = elo.update_ratings(
            ent_a[2], ent_b[2], score_a, score_b, matches_a, matches_b
        )
        datastore.update_rating(ent_a[0], new_a)
        datastore.update_rating(ent_b[0], new_b)
        datastore.record_match(ent_a[0], ent_b[0], score_a, score_b)

        console.print(
            f"[bold]Updated[/]: {ent_a[1]} → {new_a:.1f}, {ent_b[1]} → {new_b:.1f}"
        )
        last_pair, last_winner = pair_ids, (winner[0] if winner else None)


def recompute_ratings(datastore: DataStore, settings: Dict[str, Any]) -> None:
    """
    Reset ratings and replay all matches from history log.

    Parameters
    ----------
    datastore : DataStore
    settings : Dict[str, Any]
        Contains 'initial_rating', 'k_factor', and optional 'provisional_k_factor'.
    """
    logger.info("Recomputing all ratings from history")
    history = datastore.get_match_history()
    entities_info = [{"id": e[0], "name": e[1]} for e in datastore.get_all_entities()]
    datastore.reset()
    datastore.add_entities(entities_info, settings["initial_rating"])

    elo = EloSystem(settings["k_factor"], settings.get("provisional_k_factor"))
    for ent_a_id, ent_b_id, score_a, score_b, timestamp in history:
        ratings_map = {e[0]: e[2] for e in datastore.get_all_entities()}
        matches_before = {
            eid: datastore.count_matches(eid) for eid in (ent_a_id, ent_b_id)
        }
        new_a, new_b = elo.update_ratings(
            ratings_map[ent_a_id],
            ratings_map[ent_b_id],
            score_a,
            score_b,
            matches_before[ent_a_id],
            matches_before[ent_b_id],
        )
        datastore.update_rating(ent_a_id, new_a)
        datastore.update_rating(ent_b_id, new_b)
        datastore.record_match(ent_a_id, ent_b_id, score_a, score_b, timestamp)

    logger.info("Recompute complete")


def export_rankings(datastore: DataStore, output_path: str) -> None:
    """
    Export current rankings with match counts to a YAML file.

    Parameters
    ----------
    datastore : DataStore
    output_path : str
        Path to output YAML file.
    """
    logger.info(f"Exporting rankings to {output_path}")
    ents = datastore.get_all_entities()
    ranked = sorted(ents, key=lambda x: x[2], reverse=True)
    export_data = []
    for eid, name, rating in ranked:
        export_data.append(
            {
                "id": eid,
                "name": name,
                "rating": rating,
                "matches": datastore.count_matches(eid),
            }
        )
    save_yaml(export_data, output_path)
    console.print(f"[bold green]Rankings exported to[/] {output_path}")


def create_templates(directory: str) -> None:
    """
    Generate example YAML templates for things and settings.

    Parameters
    ----------
    directory : str
        Directory to output template files.
    """
    logger.info(f"Creating template files in {directory}")
    os.makedirs(directory, exist_ok=True)
    things_template = ["First Entity", "Second Entity"]
    settings_template = {
        "initial_rating": 1500,
        "k_factor": 32,
        "provisional_k_factor": 64,
    }
    save_yaml(things_template, os.path.join(directory, "things.yaml"))
    save_yaml(settings_template, os.path.join(directory, "settings.yaml"))
    console.print(
        f"[bold green]Template files created in[/] {directory}: things.yaml, settings.yaml"
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Elo Rating System CLI", formatter_class=RichHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    cmd_init = sub.add_parser(
        "init", help="Initialize the Elo system with entities and settings"
    )
    cmd_init.add_argument(
        "--things",
        required=True,
        help="Path to YAML file containing list of entities (things.yaml)",
    )
    cmd_init.add_argument(
        "--settings",
        required=True,
        help="Path to YAML file containing settings (settings.yaml)",
    )

    # play command
    cmd_play = sub.add_parser("play", help="Play a single match and update ratings")
    cmd_play.add_argument(
        "--smart",
        action="store_true",
        help="Use smart pairing based on rating difference",
    )

    # play-all command
    cmd_play_all = sub.add_parser(
        "play-all", help="Continuously play matches until user quits"
    )
    cmd_play_all.add_argument(
        "--smart",
        action="store_true",
        help="Use smart pairing based on rating difference",
    )
    cmd_play_all.add_argument(
        "--assume-win",
        action="store_true",
        help="Assume previous match winner wins again",
    )

    # recompute command
    cmd_recompute = sub.add_parser(
        "recompute", help="Recompute all ratings from match history"
    )
    cmd_recompute.add_argument(
        "--settings",
        required=True,
        help="Path to YAML file containing settings for recompute",
    )

    # export command
    cmd_export = sub.add_parser("export", help="Export current rankings to a YAML file")
    cmd_export.add_argument(
        "--output",
        default="rankings.yaml",
        help="Output path for exported rankings YAML file",
    )

    # reset command
    sub.add_parser("reset", help="Reset the database and reinitialize schema")

    # template command
    cmd_template = sub.add_parser(
        "template", help="Generate YAML template files for entities and settings"
    )
    cmd_template.add_argument(
        "--dir", required=True, help="Directory to output generated template files"
    )

    # stats command
    sub.add_parser(
        "stats", help="Display current ratings and match counts for all entities"
    )

    return parser


def main() -> None:
    """
    Entry point for CLI execution.
    """
    parser = build_parser()
    args = parser.parse_args()

    datastore = DataStore()
    settings: Dict[str, Any] = {}

    if args.command == "init":
        things = load_yaml(args.things)
        settings = load_yaml(args.settings)
        datastore.reset()
        datastore.add_entities(
            [{"id": t, "name": t} for t in things], settings["initial_rating"]
        )
        console.print("[bold green]Initialization complete.[/]")
    elif args.command == "play":
        settings = {}  # assume settings loaded earlier or defaults
        elo = EloSystem(
            settings.get("k_factor", 32), settings.get("provisional_k_factor")
        )
        play_single_match(datastore, elo, smart=args.smart)
    elif args.command == "play-all":
        settings = {}
        elo = EloSystem(
            settings.get("k_factor", 32), settings.get("provisional_k_factor")
        )
        play_continuous(datastore, elo, smart=args.smart, assume_win=args.assume_win)
    elif args.command == "recompute":
        settings = load_yaml(args.settings)
        recompute_ratings(datastore, settings)
    elif args.command == "export":
        export_rankings(datastore, args.output)
    elif args.command == "reset":
        datastore.reset()
    elif args.command == "template":
        create_templates(args.dir)
    elif args.command == "stats":
        ents = datastore.get_all_entities()
        for eid, name, rating in ents:
            count = datastore.count_matches(eid)
            console.print(f"{name}: rating={rating:.1f}, matches={count}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
