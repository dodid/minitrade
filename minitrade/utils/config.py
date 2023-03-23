"""Read and write minitrade config file

Note all config items should have default values so that an initial configuration file can 
be generated on installation.
"""

import logging
import os
import sys
from posixpath import expanduser

import yaml
from pydantic import BaseModel


class SourceConfigYahoo(BaseModel):
    proxy: str | None = None


class SourceConfig(BaseModel):
    yahoo: SourceConfigYahoo = SourceConfigYahoo()


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
    api_key: str | None
    api_secret: str | None
    sender: str | None
    mailto: str | None


class ProviderConfigTelegram(BaseModel):
    token: str | None
    chat_id: str | None
    proxy: str | None


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
            config = GlobalConfig.parse_obj(obj)
            return config
        except Exception as e:
            raise RuntimeError('Loading configuration error') from e

    def save(self, config_yaml: str = '~/.minitrade/config.yaml'):
        ''' Save minitrade configuration '''
        with open(expanduser(config_yaml), 'w') as f:
            f.write(yaml.safe_dump(self.dict()))


if 'pytest' not in sys.modules:
    try:
        config = GlobalConfig.load()
    except Exception:
        pass
else:
    assert os.path.exists(expanduser('~/.minitrade/config.pytest.yaml'))
    config = GlobalConfig.load('~/.minitrade/config.pytest.yaml')
