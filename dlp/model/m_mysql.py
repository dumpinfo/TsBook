import time
import threading
import pymysql
import app_global as ag
import conf.app_conf as conf

rdb_pool_num = 5
rdb_pool = []
rdb_pool_lock = threading.Lock()

wdb_pool_num = 5
wdb_pool = []
wdb_pool_lock = threading.Lock()

class Db_Pool_Cleaner(threading.Thread):
    def __init__(self, db_pool, db_pool_lock, duration, interval):
        threading.Thread.__init__(self)
        self.db_pool_lock = db_pool_lock
        self.db_pool = db_pool
        self.duration = duration
        self.interval = interval
        self.is_stopping = False

    def run(self):
        curr_time = time.time()
        self.db_pool_lock.acquire()
        for item in self.db_pool:
            use_time = item['use_time']
            if curr_time - use_time > self.duration:
                item['state'] = 0
                item['use_time'] = 0
        self.db_pool_lock.release()
        time.sleep(self.duration + self.interval)
        if self.is_stopping:
            return
        self.run()

def create_rdb_conn():
    return create_db_conn(conf.rdb)

def create_wdb_conn():
    return create_db_conn(conf.wdb)

def create_db_conn(db):
    return pymysql.connect(
        host = db['host'],
        port = db['port'],
        user = db['user'],
        passwd = db['passwd'],
        db = db['db'],
        charset = db['charset']
    )

def _init_db_pool(db_pool, db_pool_num, create_db_conn_func):
    state = 0
    use_time = 0
    for idx in range(db_pool_num):
        conn = create_db_conn_func()
        db_pool.append({'idx': idx, 'conn': conn, 'state': state, \
                        'use_time': use_time})

ag.rdb_pool_cleaner = Db_Pool_Cleaner(rdb_pool, rdb_pool_lock, 3600, 100)
ag.wdb_pool_cleaner = Db_Pool_Cleaner(wdb_pool, wdb_pool_lock, 3600, 100)
def init_db_pool():
    _init_db_pool(rdb_pool, rdb_pool_num, create_rdb_conn)
    _init_db_pool(wdb_pool, wdb_pool_num, create_wdb_conn)
    ag.rdb_pool_cleaner.start()
    ag.wdb_pool_cleaner.start()

def _get_db_connection(db_pool, db_pool_lock):
    db_pool_lock.acquire()
    for item in db_pool:
        if (0 == item['state']):
            item['state'] = 1
            item['use_time'] = time.time()
            db_pool_lock.release()
            return item
    db_pool_lock.release()
    return None

def close_db_connection(conn_obj):
    conn_obj['conn'].commit()
    conn_obj['state'] = 0
    conn_obj['use_time'] = 0

def get_rdb_connection():
    return _get_db_connection(rdb_pool, rdb_pool_lock)

def get_wdb_connection():
    return _get_db_connection(wdb_pool, wdb_pool_lock)

def query(sql, params):
    conn_obj = get_rdb_connection()
    conn = conn_obj['conn']
    result = query_t(conn, sql, params)
    close_db_connection(conn_obj)
    return result

def query_t(conn, sql, params):
    cursor = conn.cursor()
    cursor.execute(sql, params)
    rowcount = cursor.rowcount
    rows = cursor.fetchall()
    cursor.close()
    return (rowcount, rows)
    
def insert(sql, params):
    conn_obj = get_wdb_connection()
    conn = conn_obj['conn']
    result = insert_t(conn, sql, params)
    close_db_connection(conn_obj)
    return result

def insert_t(conn, sql, params):
    cursor = conn.cursor()
    affected_rows = cursor.execute(sql, params)
    conn.commit()
    cursor.close()
    pk = cursor.lastrowid
    return (pk, affected_rows)
    
def delete(sql, params):
    conn_obj = get_wdb_connection()
    conn = conn_obj['conn']
    result = delete_t(conn, sql, params)
    close_db_connection(conn_obj)
    return result
    
def delete_t(conn, sql, params):
    cursor = conn.cursor()
    affected_rows = cursor.execute(sql, params)
    conn.commit()
    cursor.close()
    return (0, affected_rows)
    
def update(sql, params):
    conn_obj = get_wdb_connection()
    conn = conn_obj['conn']
    result = update_t(conn, sql, params)
    close_db_connection(conn_obj)
    return result
    
def update_t(conn, sql, params):
    cursor = conn.cursor()
    affected_rows = cursor.execute(sql, params)
    conn.commit()
    cursor.close()
    return (0, affected_rows)




















