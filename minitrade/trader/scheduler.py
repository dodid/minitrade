import logging
from datetime import datetime, time

from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.job import Job
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel

from minitrade.trader import BacktestRunner, TradePlan, Trader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title='Minitrade runner')


class JobInfo(BaseModel):
    job_id: str
    job_frequency: str
    next_run_time: datetime


def schedule_trade_plan(plan: TradePlan) -> Job | None:
    ''' Schedule a trade plan'''
    if plan.enabled:
        logger.info(f'Scheduling trade plan: {plan.name}')
        trade_time = time.fromisoformat(plan.trade_time_of_day)
        job = app.scheduler.add_job(
            BacktestRunner(plan).execute,
            'cron',
            day_of_week='0-4',
            hour=trade_time.hour,
            minute=trade_time.minute,
            second=trade_time.second,
            timezone=plan.market_timezone,
            misfire_grace_time=3600,
            id=plan.id,
            name=plan.name,
            replace_existing=True
        )
        return job
    else:
        try:
            # this throws apscheduler.jobstores.base.JobLookupError if id is not found
            app.scheduler.remove_job(plan.id)
        except Exception:
            pass


def load_trade_plans() -> list[Job]:
    ''' Reload all trade plans '''
    plan_lst = TradePlan.list_plans()
    jobs = [schedule_trade_plan(plan) for plan in plan_lst]
    return [j for j in jobs if j is not None]


def load_trader() -> None:
    ''' Load trader '''
    app.scheduler.add_job(
        Trader().execute,
        'cron',
        minute='*/10',
        misfire_grace_time=3600,
        id='trader-hf7749d',
        name='trader-hf7749d',
        replace_existing=True
    )


def job_info(job: Job) -> dict:
    ''' Extract job info '''
    return {'job_id': job.id, 'job_frequency': str(job.trigger), 'next_run_time': job.next_run_time}


def create_scheduler() -> AsyncIOScheduler:
    ''' Create a scheduler instance '''
    # run job sequentially. yfinance lib may throw "Tkr {} tz already in cache" exception
    # when multiple processes run in parallel
    executors = {'default': ProcessPoolExecutor(1)}
    job_defaults = {'coalesce': True, 'max_instances': 1}
    return AsyncIOScheduler(executors=executors, job_defaults=job_defaults)


@app.on_event('startup')
def start_scheduler():
    ''' Load trade plan '''
    app.scheduler = create_scheduler()
    app.scheduler.start()
    load_trade_plans()
    load_trader()


@app.on_event('shutdown')
def shutdown_scheduler():
    ''' shutdown the scheduler '''
    app.scheduler.shutdown()


def get_plan(plan_id: str) -> TradePlan:
    plan = TradePlan.get_plan(plan_id)
    if plan:
        return plan
    else:
        raise HTTPException(404, f'TradePlan {plan_id} not found')


@app.get('/jobs', response_model=list[JobInfo])
def get_jobs():
    ''' Return the currently scheduled jobs '''
    return [job_info(job) for job in app.scheduler.get_jobs()]


@app.get('/jobs/{plan_id}', response_model=JobInfo)
def get_jobs_by_id(plan_id: str):
    ''' Return the specific job '''
    job = app.scheduler.get_job(job_id=plan_id)
    return job_info(job) if job else Response(status_code=204)


@app.post('/jobs', response_model=list[JobInfo])
def post_jobs():
    ''' Reschedule all trade plans '''
    return [job_info(job) for job in load_trade_plans()]


@app.put('/jobs/{plan_id}', response_model=JobInfo)
def put_jobs(plan=Depends(get_plan)):
    ''' Reschedule a single trade plan '''
    job = schedule_trade_plan(plan)
    return job_info(job) if job else Response(status_code=204)


@app.delete('/jobs/{plan_id}')
def delete_jobs(plan_id: str):
    ''' Unschedule a single trade plan '''
    try:
        app.scheduler.remove_job(plan_id)
        return Response(status_code=204)
    except Exception:
        raise HTTPException(status_code=404)
