"""
database.py
SQLite database management for caching and API usage statistics.
"""

import sqlite3
import hashlib
import os
from datetime import datetime
from config import DB_PATH


def init_db():
    """Initializes the SQLite database schemas."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table for caching generated images and prompts, now including model and resolution
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            image_path TEXT NOT NULL,
            model TEXT DEFAULT 'Unknown',
            resolution TEXT DEFAULT 'Unknown',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table for tracking API key usage and costs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_stats (
            key_hash TEXT PRIMARY KEY,
            total_images INTEGER DEFAULT 0,
            total_cost_cents INTEGER DEFAULT 0,
            current_month TEXT NOT NULL,
            monthly_images INTEGER DEFAULT 0,
            monthly_cost_cents INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def cache_image(prompt: str, image_path: str, model: str, resolution: str):
    """Saves a generated image path and its metadata to the cache."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO image_cache (prompt, image_path, model, resolution) VALUES (?, ?, ?, ?)",
        (prompt, image_path, model, resolution),
    )
    conn.commit()
    conn.close()


def get_cached_history() -> list:
    """Retrieves all cached images and their parameters, ordered by newest first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Fetching columns: image_path, prompt, model, resolution, timestamp
    cursor.execute(
        "SELECT image_path, prompt, model, resolution, timestamp "
        "FROM image_cache "
        "ORDER BY timestamp DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def clear_cache():
    """Wipes the local image cache and deletes physical files if necessary."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM image_cache")
    rows = cursor.fetchall()

    for row in rows:
        path = row[0]
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    cursor.execute("DELETE FROM image_cache")
    conn.commit()
    conn.close()


def update_stats(api_key: str, images_generated: int, cost_cents: int):
    if not api_key:
        return

    hashed = hash_key(api_key)
    current_month = datetime.now().strftime("%Y-%m")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT current_month, monthly_images, monthly_cost_cents FROM api_stats WHERE key_hash=?",
        (hashed,),
    )
    row = cursor.fetchone()

    if row is None:
        cursor.execute(
            "INSERT INTO api_stats ("
            "key_hash, "
            "total_images, "
            "total_cost_cents, "
            "current_month, "
            "monthly_images, "
            "monthly_cost_cents) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                hashed,
                images_generated,
                cost_cents,
                current_month,
                images_generated,
                cost_cents,
            ),
        )
    else:
        db_month, monthly_imgs, monthly_cost = row
        if db_month != current_month:
            monthly_imgs = 0
            monthly_cost = 0

        cursor.execute(
            """
            UPDATE api_stats
            SET total_images = total_images + ?,
                total_cost_cents = total_cost_cents + ?,
                current_month = ?,
                monthly_images = ? + ?,
                monthly_cost_cents = ? + ?
            WHERE key_hash = ?
        """,
            (
                images_generated,
                cost_cents,
                current_month,
                monthly_imgs,
                images_generated,
                monthly_cost,
                cost_cents,
                hashed,
            ),
        )

    conn.commit()
    conn.close()


def get_stats(api_key: str) -> dict:
    if not api_key:
        return {"total_img": 0, "total_cost": 0, "monthly_img": 0, "monthly_cost": 0}

    hashed = hash_key(api_key)
    current_month = datetime.now().strftime("%Y-%m")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT total_images, total_cost_cents, current_month, monthly_images, monthly_cost_cents "
        "FROM api_stats "
        "WHERE key_hash=?",
        (hashed,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"total_img": 0, "total_cost": 0, "monthly_img": 0, "monthly_cost": 0}

    tot_img, tot_cost, db_month, mon_img, mon_cost = row

    if db_month != current_month:
        mon_img, mon_cost = 0, 0

    return {
        "total_img": tot_img,
        "total_cost": tot_cost,
        "monthly_img": mon_img,
        "monthly_cost": mon_cost,
    }
