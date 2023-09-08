---
hide:
  - toc
---

# Minitrade

**Minitrade** is a personal trading system that supports both strategy backtesting and automated order execution. It builds on top of [Backtesting.py](https://github.com/kernc/backtesting.py), and provides enhanced features such as:

- **Multi-asset rebalancing strategy backtest**
- Automated strategy execution and order submission
- Web-based management UI
- Notification and control on mobile

With Minitrade, you can set up a private and fully automated stock trading system for as low as $5/mo.

## Installation

    $ pip install minitrade
    $ minitrade init

Minitrade requires `python=3.10.*`. Check out [Installation](install.md) for more details.

## Usage

### Backtesting

| ![Minitrade backtesting](https://imgur.com/YkLPeTv) |
| --------------------------------------------------- |

### Trading

| ![Minitrade web UI - history](<https://imgur.com/ittnlk7>) |
| ---------------------------------------------------------- |
| ![Minitrade web UI - orders](<https://imgur.com/2DAZ2W1>)  |

## Limitations

As a backtesting framework:

- Multi-asset strategy only supports long positions and market order. 

As a trading system:

- Tested only on Linux
- Support only daily bar
- Support only long positions
- Support only Interactive Brokers

## Contributing

Check out [how to contribute](contributing.md).

## License

[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)
