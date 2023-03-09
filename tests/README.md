## Note on running tests

Do not run tests in a production environment. It will erase the installed strategies.

To run in a dev environment:
1. Copy `~/.minitrade/config.yaml` to `~/.minitrade/config.pytest.yaml`.
2. Edit `config.pytest.yaml` to specify different ports for `brokers.ib.gateway_admin_port` and `scheduler.port`.
3. Open a shell and set up the environment variables for test IB account. **Use a PAPER account.**
   ```
   $ export IB_TEST_USERNAME=xxxx
   $ export IB_TEST_PASSWORD=xxxx
4. Run from project root.
   ```
   $ pytest tests
   ```