[![Static Badge](https://img.shields.io/badge/Documentation-blue)](https://dodid.github.io/minitrade/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/dodid/minitrade/test_code.yml?branch=main)](https://github.com/dodid/minitrade/actions)
[![Backtesting on PyPI](https://img.shields.io/pypi/v/minitrade.svg?color=blue)](https://pypi.org/project/minitrade)
[![PyPI downloads](https://img.shields.io/pypi/dm/minitrade.svg?color=skyblue)](https://pypi.org/project/minitrade)


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

Minitrade requires `python=3.10.*`. Check out [Installation](https://dodid.github.io/minitrade/install/) for more details.

## Usage

### Backtesting

| ![Minitrade backtesting](https://imgur.com/YkLPeTv.jpg) |
| ------------------------------------------------------- |

### Trading

| ![Minitrade web UI - history](<https://imgur.com/ittnlk7.png>) |
| -------------------------------------------------------------- |
| ![Minitrade web UI - orders](<https://imgur.com/2DAZ2W1.png>)  |

See more in [Documentation](https://dodid.github.io/minitrade/).

## Limitations

As a backtesting framework:

- Multi-asset strategy only supports long positions and market order. 

As a trading system:

- Tested only on Linux
- Support only daily bar
- Support only long positions
- Support only Interactive Brokers

## Contributing

Check out [how to contribute](CONTRIBUTING.md).

## License

[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)
