# Licenses and Attributions

This project (`humanlike-chess`) is developed as an open research and human-like chess simulation system. It integrates, references, and interacts with several upstream open-source tools, models, and datasets.

---

## 1. Project License
- **License**: GNU Affero General Public License v3.0 (AGPL-3.0)
- **Rationale**: Direct integration with Maia-3 and `lichess-bot` (both AGPLv3).
- **Non-Commercial Boundary**: Due to potential integration with ChessMimic (PolyForm Noncommercial 1.0.0), this repository and all artifacts must remain strictly non-commercial.

---

## 2. Upstream Components and Dependencies

### Maia-3
- **Repository**: [https://github.com/CSSLab/maia3](https://github.com/CSSLab/maia3)
- **Weights**: [https://huggingface.co/collections/MaiaChess/maia3](https://huggingface.co/collections/MaiaChess/maia3) (e.g. `UofTCSSLab/Maia3-79M`, `UofTCSSLab/Maia3-5M`)
- **License**: AGPL-3.0 (verified on repository and Hugging Face model cards).
- **Citation**:
  ```bibtex
  @inproceedings{monroe2026chessformer,
    title={Chessformer: A Unified Architecture for Chess Modeling},
    author={Daniel Monroe and George Eilender and Philip Chalmers and Zhenwei Tang and Ashton Anderson},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=2ltBRzEHyd}
  }
  ```

### lichess-bot
- **Repository**: [https://github.com/lichess-bot-devs/lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)
- **License**: AGPLv3.

### Lichess Open Database
- **Source**: [https://database.lichess.org](https://database.lichess.org)
- **License**: Creative Commons CC0 1.0 Universal (Public Domain Dedication).
- **Attribution**: Games played on Lichess.org, exported monthly under CC0.

### ChessMimic (Phase 6 reference / optional)
- **Repository**: [https://github.com/thomasj02/1e4_ai](https://github.com/thomasj02/1e4_ai)
- **Paper**: arXiv:2606.04473
- **License**: PolyForm Noncommercial License 1.0.0.
- **Notice**: Must not be used for commercial purposes.

### python-chess
- **Repository**: [https://github.com/niklasf/python-chess](https://github.com/niklasf/python-chess)
- **License**: GPL-3.0-or-later.
