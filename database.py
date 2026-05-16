import pickle
import os
from config import DB_FILE, BACKBONE


def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'rb') as f:
            db = pickle.load(f)

        # ── Check backbone compatibility ─────────────────────────
        saved_backbone = db.get('__backbone__', None)
        # Filter out metadata keys for display
        people = {k: v for k, v in db.items() if not k.startswith('__')}

        if saved_backbone is not None and saved_backbone != BACKBONE:
            print(f"⚠  Backbone changed: '{saved_backbone}' → '{BACKBONE}'")
            print(f"   Clearing incompatible database ({len(people)} people removed).")
            db = {'__backbone__': BACKBONE}
            save_database(db)
            return db

        if saved_backbone is None:
            # Legacy DB without backbone tag — check dim compatibility
            print(f"⚠  Legacy database (no backbone tag). Re-enroll recommended.")
            db['__backbone__'] = BACKBONE
            save_database(db)

        print(f"Loaded {len(people)} people: {list(people.keys())}")
        return db

    print("Empty database")
    return {'__backbone__': BACKBONE}


def save_database(database):
    with open(DB_FILE, 'wb') as f:
        pickle.dump(database, f)
    people = {k: v for k, v in database.items() if not k.startswith('__')}
    print(f"Saved: {list(people.keys())}")


def add_person(database, name, embedding):
    if name not in database:
        database[name] = []
    database[name].append(embedding.flatten())
    save_database(database)
    people_count = len(database[name])
    print(f"Enrolled: {name} ({people_count} samples)")


def delete_person(database, name):
    """Remove a person from the database by name."""
    if name in database and not name.startswith('__'):
        del database[name]
        save_database(database)
        people = {k: v for k, v in database.items() if not k.startswith('__')}
        print(f"Deleted '{name}' from database.")
        print(f"Remaining: {list(people.keys()) if people else '(empty)'}")
        return True
    else:
        people = {k: v for k, v in database.items() if not k.startswith('__')}
        print(f"'{name}' not found in database.")
        print(f"Available: {list(people.keys()) if people else '(empty)'}")
        return False


def list_people(database):
    """List all people in the database."""
    people = {k: v for k, v in database.items() if not k.startswith('__')}
    if not people:
        print("Database is empty.")
    else:
        print(f"People in database ({len(people)}):")
        for name, vectors in people.items():
            print(f"  - {name} ({len(vectors)} samples)")


def get_people(database):
    """Return dict of people only (no metadata keys)."""
    return {k: v for k, v in database.items() if not k.startswith('__')}