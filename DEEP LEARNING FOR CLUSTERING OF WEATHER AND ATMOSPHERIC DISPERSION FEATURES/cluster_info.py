 from dataset_utils import load_single

clust_obj = load_single('<path to clustering object>')
print clust_obj._desc_date

from datetime import datetime
def reconstruct_date(date_str, dot_nc=False):
    if dot_nc:
        date = datetime.strptime(
            date_str.split('.')[0], '%Y-%m-%d_%H:%M:%S')
    else:
        date = datetime.strptime(date_str, '%Y-%m-%d_%H:%M:%S')
    return datetime.strftime(date, '%y-%m-%d-%H')

q = []
for i in clust_obj._desc_date:
    q.append(reconstruct_date(i))



