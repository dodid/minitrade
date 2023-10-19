[![Static Badge](https://img.shields.io/badge/Documentation-blue)](https://dodid.github.io/minitrade/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/dodid/minitrade/test_code.yml?branch=main)](https://github.com/dodid/minitrade/actions)
[![Backtesting on PyPI](https://img.shields.io/pypi/v/minitrade.svg?color=blue)](https://pypi.org/project/minitrade)
[![PyPI downloads](https://img.shields.io/pypi/dm/minitrade.svg?color=skyblue)](https://pypi.org/project/minitrade)


# Minitrade - Simplifying Personal Trading

**Minitrade** is a personal trading system that combines strategy backtesting and automated order execution. Leveraging the power of [Backtesting.py](https://github.com/kernc/backtesting.py), Minitrade offers an array of enhanced features:

- **Multi-asset rebalancing strategy backtest**: Explore and optimize your trading strategies across various assets.
- Automated strategy execution and order submission: Seamlessly execute your trading strategies and submit orders automatically.
- Web-based management UI: Access a user-friendly web interface to manage and monitor your trading activities.
- Streamlined mobile notifications and control: Stay informed and in control with mobile notifications for important trading events.

With Minitrade, setting up your private and fully automated stock trading system is easy and affordable.

## Installation

    $ pip install minitrade
    $ minitrade init

Minitrade requires `python >= 3.10`. `"minitrade init"` is only necessary if you use Minitrade for trading. Run it again after every Minitrade upgrade. For detailed installation instructions, refer to the [Installation](https://dodid.github.io/minitrade/install/) section on the website.

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

Please be aware of the following limitations:

- Multi-asset strategies currently support long positions and market orders only.
- Minitrade has been tested on Linux platforms.
- Daily bar data is currently supported.
- Interactive Brokers is the supported broker.

## Contributing

Check out the guidelines on [how to contribute](CONTRIBUTING.md) to the project.

## License

[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)
