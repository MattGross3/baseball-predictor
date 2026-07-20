import datetime as dt

from features.injury_features import replay_il_transactions


def _txn(date: str, player_id: int, description: str) -> dict:
    return {"date": date, "person": {"id": player_id}, "description": description}


class TestReplayIlTransactions:
    def test_placed_on_il_shows_up(self):
        txns = [_txn("2026-06-14", 608070, "Cleveland Guardians placed 3B José Ramírez on the 10-day injured list.")]
        assert replay_il_transactions(txns, dt.date(2026, 7, 1)) == {608070}

    def test_activated_removes_from_il(self):
        txns = [
            _txn("2026-06-14", 672356, "Cleveland Guardians placed SS Gabriel Arias on the 60-day injured list."),
            _txn("2026-07-01", 672356, "Cleveland Guardians activated SS Gabriel Arias from the 60-day injured list."),
        ]
        assert replay_il_transactions(txns, dt.date(2026, 7, 10)) == set()

    def test_transactions_on_or_after_as_of_date_are_not_counted(self):
        # A long-term injury that started well before as_of_date IS
        # counted (the whole point of this being date-reconstructable,
        # not a live snapshot) - this test is about the boundary itself.
        txns = [_txn("2026-06-14", 608070, "placed 3B José Ramírez on the 10-day injured list.")]
        assert replay_il_transactions(txns, dt.date(2026, 6, 14)) == set()
        assert replay_il_transactions(txns, dt.date(2026, 6, 15)) == {608070}

    def test_long_term_injury_still_counted_well_after_placement(self):
        # This is the bug this whole module exists to avoid: a player out
        # for months shouldn't "age out" of being counted as injured just
        # because the placement transaction is old.
        txns = [_txn("2026-04-01", 608070, "placed 3B José Ramírez on the 60-day injured list.")]
        assert replay_il_transactions(txns, dt.date(2026, 7, 20)) == {608070}

    def test_unrelated_transactions_are_ignored(self):
        txns = [
            _txn("2026-06-01", 111, "Cleveland Guardians optioned RHP Foo Bar to Columbus Clippers."),
            _txn("2026-06-02", 222, "Cleveland Guardians traded RF Baz Qux to Chicago White Sox."),
        ]
        assert replay_il_transactions(txns, dt.date(2026, 7, 1)) == set()

    def test_transactions_out_of_order_are_still_replayed_correctly(self):
        txns = [
            _txn("2026-07-01", 555, "activated LHP Foo from the 15-day injured list."),
            _txn("2026-06-01", 555, "placed LHP Foo on the 15-day injured list."),
        ]
        assert replay_il_transactions(txns, dt.date(2026, 6, 15)) == {555}
        assert replay_il_transactions(txns, dt.date(2026, 7, 15)) == set()
