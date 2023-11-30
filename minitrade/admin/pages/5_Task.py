from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

import streamlit as st
from apscheduler.triggers.cron import CronTrigger

from minitrade.trader import TaskManager, TaskPlan, TaskRunner

st.set_page_config(page_title='Task', layout='wide')


def show_task_uploader():
    ''' Create task uploader control '''
    uploaded_file = st.sidebar.file_uploader('Upload task file', type=['py'], accept_multiple_files=False)
    if uploaded_file is not None:
        try:
            filename = uploaded_file.name.replace(' ', '_')
            TaskManager.save(filename, uploaded_file.getvalue().decode("utf-8"))
            return filename
        except Exception as e:
            st.error(e)


def confirm_delete_task(plan: TaskPlan) -> None:
    def confirm_delete():
        if st.session_state.delete_confirm_textinput == plan.name:
            plan.delete()
            TaskManager.delete(plan.task_file)
    st.text_input(f'Type "{plan.name}" and press Enter to delete',
                  on_change=confirm_delete, key='delete_confirm_textinput')


def run_task_once(plan: TaskPlan) -> None:
    log = TaskRunner(plan).execute()
    if log is not None and not log.error:
        st.success(f'Run {log.id} finished successfully')
    else:
        st.error(f'Run failed')


def show_task_header_and_controls(plan: TaskPlan):
    c1, c2, c3, c4, c5 = st.columns([4, 1, 1, 1, 1])
    c1.subheader(plan.name)
    c2.button('Refresh')
    c3.button('Run Once', on_click=lambda: run_task_once(plan))
    c4.button('Disable' if plan.enabled else 'Enable', key='taskplan_onoff',
              on_click=lambda: plan.enable(not plan.enabled), disabled=not plan.schedule)
    if c5.button('Delete', type="primary"):
        confirm_delete_task(plan)


def show_task_selector(uploaded_file: str) -> str | None:
    names = [p.name for p in TaskPlan.list()]
    selected_index = names.index(uploaded_file) if uploaded_file else 0
    return st.sidebar.radio('Task', names, index=selected_index)


def show_schedule_editor(tab, plan: TaskPlan):
    show_task_plan_status(tab, plan)
    zones = [None] + sorted(list(available_timezones()))
    timezone = tab.selectbox('Schedule timezone', zones, index=zones.index(
        plan.timezone) if plan.timezone in zones else 0)
    help_msg = 'Define task schedule in crontab format. You can have multiple entries in separate lines. See https://crontab.guru/ for help.'
    schedule = tab.text_area('Schedule in crontab format', plan.schedule.replace(',', '\n') if plan.schedule else '',
                             help=help_msg, placeholder='e.g. 0 9 * * 1-5')
    notification = {
        'telegram': tab.selectbox(
            'Telegram notification', ['N', 'E', 'A'],
            format_func=lambda _: {'N': 'None', 'E': 'Error only', 'A': 'Always'}[_],
            index='NEA'.index(plan.notification.get('telegram', 'N') if plan.notification else 'N')),
        'email': tab.selectbox(
            'Email notification', ['N', 'E', 'A'],
            format_func=lambda _: {'N': 'None', 'E': 'Error only', 'A': 'Always'}[_],
            index='NEA'.index(plan.notification.get('email', 'N') if plan.notification else 'N')),
    }
    if tab.button('Save'):
        try:
            if not timezone:
                raise AttributeError('Timezone cannot be empty.')
            datetime.now(ZoneInfo(timezone))
            crontab = []
            for line in schedule.splitlines():
                line = line.strip()
                if line:
                    m, h, d, M, w = line.strip().split(' ')
                    CronTrigger(minute=m, hour=h, day=d, month=M, day_of_week=w, timezone=timezone)
                    crontab.append(line)
            plan.timezone = timezone
            plan.schedule = ','.join(crontab)
            plan.notification = notification
            plan.save()
            tab.success('Schedule saved')
        except AttributeError as e:
            tab.error(e)
        except ZoneInfoNotFoundError as e:
            tab.error(f'Invalid timezone "{timezone}": {e}')
        except ValueError as e:
            tab.error(f'Invalid crontab entry "{line}": {e}')


def show_task_plan_status(tab, plan: TaskPlan) -> None:
    with tab:
        job = plan.jobinfo()
        if job is None:
            if plan.enabled:
                st.warning('State inconsistency detected: task is enabled but not scheduled')
            else:
                st.warning('Task is disabled and not scheduled')
        else:
            if plan.enabled:
                st.success(f'Task is scheduled to run at {job["next_run_time"]}')
            else:
                st.success(
                    f'State inconsistenccy detected: task is disabled but is scheduled to run at {job["next_run_time"]}')


def show_run_history(tab, plan: TaskPlan) -> None:
    with tab:
        for log in plan.list_logs():
            log_status = '❌' if log.error else '✅'
            log_time = log.log_time.replace(tzinfo=ZoneInfo('UTC')).astimezone(
                ZoneInfo(plan.timezone)).strftime('%Y-%m-%d %H:%M:%S')
            label = f'{log_status} {log_time} [exit {log.return_value}]'
            with st.expander(label):
                st.caption(f'Log - stderr')
                if log.error:
                    st.text(log.stderr)
                st.caption(f'Log - stdout')
                if log.stdout:
                    st.text(log.stdout)


uploaded_file = show_task_uploader()
pid = show_task_selector(uploaded_file)
if pid:
    plan = TaskPlan.get_plan(pid)
    show_task_header_and_controls(plan)
    t1, t2, t3, t4 = st.tabs(['Config', 'Run History', 'Code', 'Settings'])
    show_schedule_editor(t1, plan)
    show_run_history(t2, plan)
    t3.code(TaskManager.read(plan.task_file), language='python')
    t4.write(asdict(plan))
