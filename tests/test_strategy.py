from .fixture import *


def test_strategy_manager(clean_strategy):
    st_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'strategy')
    for file in os.listdir(st_dir):
        if not file.endswith('.py'):
            continue
        with open(os.path.join(st_dir, file), 'r') as f:
            content = f.read()
        StrategyManager.save(file, content)
        assert file in StrategyManager.list()
        strategy = StrategyManager.load(file)
        assert issubclass(strategy, Strategy)
    for file in StrategyManager.list():
        StrategyManager.delete(file)
        st_loc = os.path.join(os.path.expanduser(f'~/.minitrade/strategy'), file)
        assert os.path.exists(st_loc) == False
