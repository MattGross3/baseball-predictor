"""add prediction metadata and lookup indexes

Revision ID: f0e6f7d31a3c
Revises: a2f2c99f48b8
Create Date: 2026-07-20 13:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f0e6f7d31a3c'
down_revision: Union[str, Sequence[str], None] = 'a2f2c99f48b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('predictions', sa.Column('predicted_side', sa.String(length=20), nullable=True))
    op.add_column('predictions', sa.Column('predicted_home_value', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('predicted_away_value', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('home_probability', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('away_probability', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('market_home_probability', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('market_away_probability', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('confidence', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('actual_outcome', sa.String(length=30), nullable=True))
    op.add_column('predictions', sa.Column('target_unit', sa.String(length=40), nullable=True))
    op.create_index('ix_predictions_game_target_created', 'predictions', ['game_id', 'target_type', 'created_at'], unique=False)
    op.create_index('ix_predictions_target_created', 'predictions', ['target_type', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_predictions_target_created', table_name='predictions')
    op.drop_index('ix_predictions_game_target_created', table_name='predictions')
    op.drop_column('predictions', 'target_unit')
    op.drop_column('predictions', 'actual_outcome')
    op.drop_column('predictions', 'confidence')
    op.drop_column('predictions', 'market_away_probability')
    op.drop_column('predictions', 'market_home_probability')
    op.drop_column('predictions', 'away_probability')
    op.drop_column('predictions', 'home_probability')
    op.drop_column('predictions', 'predicted_away_value')
    op.drop_column('predictions', 'predicted_home_value')
    op.drop_column('predictions', 'predicted_side')
