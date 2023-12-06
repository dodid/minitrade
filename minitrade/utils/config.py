"""Read and write minitrade config file

Note all config items should have default values so that an initial configuration file can 
be generated on installation.
"""

import os
import sys
from posixpath import expanduser

import yaml
from pydantic import BaseModel

minitrade_root = expanduser('~/.minitrade')


class SourceConfigYahoo(BaseModel):
    proxy: str | None = None


class SourceConfigEODHD(BaseModel):
    api_key: str | None = None


class SourceConfigTwelveData(BaseModel):
    api_key: str | None = None


class SourceConfigAlpaca(BaseModel):
    api_key: str | None = None
    api_secret: str | None = None


class SourceConfigTiingoData(BaseModel):
    api_key: str | None = None


class SourceConfigInteractiveBrokers(BaseModel):
    account: str | None = None


class SourceConfigFred(BaseModel):
    api_key: str | None = None


class SourceConfig(BaseModel):
    yahoo: SourceConfigYahoo = SourceConfigYahoo()
    eodhd: SourceConfigEODHD = SourceConfigEODHD()
    twelvedata: SourceConfigTwelveData = SourceConfigTwelveData()
    alpaca: SourceConfigAlpaca = SourceConfigAlpaca()
    tiingo: SourceConfigTiingoData = SourceConfigTiingoData()
    ib: SourceConfigInteractiveBrokers = SourceConfigInteractiveBrokers()
    fred: SourceConfigFred = SourceConfigFred()


class BrokerConfigIB(BaseModel):
    gateway_admin_host: str = '127.0.0.1'
    gateway_admin_port: int = 6667
    gateway_admin_log_level: str = 'info'


class BrokerConfig(BaseModel):
    ib: BrokerConfigIB = BrokerConfigIB()


class SchedulerConfig(BaseModel):
    host: str = '127.0.0.1'
    port: int = 6666
    log_level: str = 'info'


class ProviderConfigMailjet(BaseModel):
    api_key: str | None = None
    api_secret: str | None = None
    sender: str | None = None
    mailto: str | None = None


class ProviderConfigTelegram(BaseModel):
    token: str | None = None
    chat_id: str | None = None
    proxy: str | None = None


class ProviderConfig(BaseModel):
    mailjet: ProviderConfigMailjet = ProviderConfigMailjet()
    telegram: ProviderConfigTelegram = ProviderConfigTelegram()


class GlobalConfig(BaseModel):
    sources: SourceConfig = SourceConfig()
    brokers: BrokerConfig = BrokerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    providers: ProviderConfig = ProviderConfig()

    @staticmethod
    def load(config_yaml: str = '~/.minitrade/config.yaml'):
        ''' Load minitrade configuration '''
        try:
            with open(expanduser(config_yaml), 'r') as f:
                yml = f.read()
            obj = yaml.safe_load(yml)
            config = GlobalConfig.model_validate(obj)
            return config
        except Exception as e:
            raise RuntimeError('Loading configuration error') from e

    def save(self, config_yaml: str = '~/.minitrade/config.yaml'):
        ''' Save minitrade configuration '''
        with open(expanduser(config_yaml), 'w') as f:
            f.write(yaml.safe_dump(self.model_dump()))

    @staticmethod
    def upgrade():
        ''' Upgrade configuration file '''
        try:
            config = GlobalConfig.load()
        except Exception:
            config = GlobalConfig()
        config.save()


if 'pytest' not in sys.modules:
    try:
        config = GlobalConfig.load()
    except Exception:
        pass
else:
    assert os.path.exists(expanduser('~/.minitrade/config.pytest.yaml'))
    config = GlobalConfig.load('~/.minitrade/config.pytest.yaml')
