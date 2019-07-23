import hashlib
import app_global as ag

def validate_req(params):
    user_id = params.get('user_id', '0')
    req_id = params.get('req_id', '0')
    mac = params.get('mac', 'x')
    raw_str = 'user_id=' + user_id + '&req_id=' + req_id + \
               '&shared_key=' + ag.shared_key
    new_mac = hashlib.sha1(raw_str.encode('utf8')).hexdigest()
    if new_mac == mac:
        return True
    else:
        return False


