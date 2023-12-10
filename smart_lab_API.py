import os 
# os.system("pip install requests'")
import requests


def get_value(value_name, json_data):
    val = json_data.get('data',{}).get(value_name, {}).get('values')
    if len(val):
        val = val[-1]
    else:
        val = None
    return val

def get_values_for_ticket(ticket):
    r = requests.get(f'http://localhost:8000/{ticket}/q?full_info=false&include_data=any').json()
    values = ['p_e', 'roa', 'roe', 'p_e', 'p_s', 'p_b_v', 'ev_ebitda', 'i_r_r', 'revenue', 'ebitda', 'capex', 'opex', 'e_v', 'eps']
    results = {}
    for val in values:
        results[val] = get_value(value_name=val,  json_data = r)
    return results
