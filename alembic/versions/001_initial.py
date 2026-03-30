"""001_initial

Initial schema creation — creates all five core tables:
  - users
  - oauth_tokens
  - conversations
  - pending_actions
  - audit_logs

Revision ID: 001_initial
Revises    : (none — first migration)
Create Date: 2026-03-01 00:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# ── Revision identifiers ──────────────────────────────────────────────────────
revision: str = "001_initial"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # ── users ─────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "telegram_user_id",
            sa.BigInteger(),
            nullable=False,
            comment="Telegram numeric user-id — immutable after registration",
        ),
        sa.Column(
            "ms_user_id",
            sa.String(256),
            nullable=True,
            comment="Microsoft OID returned by /me endpoint after OAuth",
        ),
        sa.Column(
            "ms_email",
            sa.String(512),
            nullable=True,
            comment="Primary SMTP address from Microsoft Graph",
        ),
        sa.Column(
            "display_name",
            sa.String(256),
            nullable=True,
            comment="Human-readable name (Telegram or MS display name)",
        ),
        sa.Column(
            "timezone",
            sa.String(64),
            server_default="Asia/Dubai",
            nullable=False,
            comment="IANA timezone identifier, e.g. 'Asia/Dubai' or 'Europe/London'",
        ),
        sa.Column(
            "preferences",
            sa.JSON(),
            server_default="{}",
            nullable=False,
            comment="Per-user preferences: notifications, default calendar, language, etc.",
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
            comment="False = soft-deleted or banned",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Row-creation timestamp (UTC)",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Last-modification timestamp (UTC)",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("telegram_user_id", name="uq_users_telegram_user_id"),
    )
    op.create_index(
        "ix_users_telegram_user_id",
        "users",
        ["telegram_user_id"],
        unique=True,
    )

    # ── oauth_tokens ──────────────────────────────────────────────────────
    op.create_table(
        "oauth_tokens",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
            comment="Owner; row is deleted when the parent user is deleted",
        ),
        sa.Column(
            "encrypted_access_token",
            sa.Text(),
            nullable=False,
            comment="Fernet-encrypted MS access token — never store plaintext",
        ),
        sa.Column(
            "encrypted_refresh_token",
            sa.Text(),
            nullable=False,
            comment="Fernet-encrypted MS refresh token — never store plaintext",
        ),
        sa.Column(
            "token_type",
            sa.String(32),
            server_default="Bearer",
            nullable=False,
            comment="Token type as returned by the OAuth server, typically 'Bearer'",
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="UTC expiry time of the access token",
        ),
        sa.Column(
            "scope",
            sa.Text(),
            nullable=True,
            comment="Space-separated OAuth scopes granted by the authorisation server",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Row-creation timestamp (UTC)",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Last-modification timestamp (UTC)",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_oauth_tokens"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_oauth_tokens_user_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_oauth_tokens_user_id", "oauth_tokens", ["user_id"])

    # ── conversations ─────────────────────────────────────────────────────
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
            comment="One active conversation per user; cascade-deleted with the user",
        ),
        sa.Column(
            "state",
            sa.String(64),
            server_default="idle",
            nullable=False,
            comment="Current dialogue state: idle | awaiting_clarification | awaiting_approval | processing",
        ),
        sa.Column(
            "last_intent",
            sa.String(64),
            nullable=True,
            comment="Most recently resolved NLU intent label, e.g. 'create_event'",
        ),
        sa.Column(
            "context",
            sa.JSON(),
            server_default="{}",
            nullable=False,
            comment="Multi-turn context bag: extracted slots, clarification questions, etc.",
        ),
        sa.Column(
            "last_message_id",
            sa.BigInteger(),
            nullable=True,
            comment="Telegram message_id of the last bot reply; used for editMessageText",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Last-modification timestamp (UTC)",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
        sa.UniqueConstraint("user_id", name="uq_conversations_user_id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_conversations_user_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"], unique=True)

    # ── pending_actions ───────────────────────────────────────────────────
    op.create_table(
        "pending_actions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "action_id",
            sa.String(64),
            nullable=False,
            comment="Client-generated UUID; used in Telegram callback_data and URL tokens",
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
            comment="Owning user; row is deleted when the parent user is deleted",
        ),
        sa.Column(
            "intent",
            sa.String(64),
            nullable=False,
            comment="NLU intent that triggered this action, e.g. 'send_email'",
        ),
        sa.Column(
            "payload",
            sa.JSON(),
            nullable=False,
            comment="Full action specification: tool name, resolved arguments, preview text",
        ),
        sa.Column(
            "status",
            sa.String(32),
            server_default="pending",
            nullable=False,
            comment="Life-cycle state: pending | approved | rejected | expired | executed | failed",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Timestamp when the action was queued for approval",
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Deadline for user approval",
        ),
        sa.Column(
            "executed_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Timestamp when the action was dispatched to the Microsoft Graph API",
        ),
        sa.Column(
            "result_meta",
            sa.JSON(),
            nullable=True,
            comment="Outcome metadata written after execution",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_pending_actions"),
        sa.UniqueConstraint("action_id", name="uq_pending_actions_action_id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_pending_actions_user_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_pending_actions_action_id", "pending_actions", ["action_id"], unique=True)
    op.create_index("ix_pending_actions_user_id", "pending_actions", ["user_id"])
    op.create_index("ix_pending_actions_status", "pending_actions", ["status"])
    op.create_index(
        "ix_pending_actions_status_expires_at",
        "pending_actions",
        ["status", "expires_at"],
    )

    # ── audit_logs ────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "correlation_id",
            sa.String(64),
            nullable=False,
            comment="Request-scoped trace ID for correlating log entries in a single flow",
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=True,
            comment="Owning user; SET NULL on delete so audit history is retained",
        ),
        sa.Column(
            "action_id",
            sa.String(64),
            nullable=True,
            comment="pending_actions.action_id value reference (not a hard FK)",
        ),
        sa.Column(
            "tool",
            sa.String(128),
            nullable=False,
            comment="Fully-qualified tool / Graph API operation, e.g. 'graph.mail.send'",
        ),
        sa.Column(
            "request_meta",
            sa.JSON(),
            nullable=False,
            comment="Sanitised request snapshot: endpoint, method, payload summary",
        ),
        sa.Column(
            "result_meta",
            sa.JSON(),
            nullable=False,
            comment="Response snapshot: HTTP status, relevant fields, error codes",
        ),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            comment="Outcome: success | failure | error",
        ),
        sa.Column(
            "duration_ms",
            sa.Integer(),
            nullable=True,
            comment="Wall-clock time of the external API call in milliseconds",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Immutable creation timestamp (UTC); rows must never be updated",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_audit_logs"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_audit_logs_user_id",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_audit_logs_correlation_id", "audit_logs", ["correlation_id"])
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_action_id", "audit_logs", ["action_id"])


def downgrade() -> None:
    # Drop in reverse dependency order (children before parents).
    op.drop_table("audit_logs")
    op.drop_table("pending_actions")
    op.drop_table("conversations")
    op.drop_table("oauth_tokens")
    op.drop_table("users")
