import matplotlib.pyplot as plt
#import tabulate
import json

log_dir = 'log/'
experiment = ['baseline', 'N', 'GC', 'BN', 'R2', 'AA', 'AC'] # baseline, BN, GC, N, AA, AC
versions = '100'#['100','200','300','400']


results = []
loss = []
acc = []

for e in experiment:
    xp_path = log_dir+e+versions
    with open(xp_path + '/results.json') as f_r:
        results.append(json.loads(f_r.read()))
    with open(xp_path + '/visdom.log') as f_r:
        dat=f_r.read().split(']\n')
        loss.append(json.loads(dat[0][11:]))
        acc.append(json.loads(dat[1][11:]))

fig, ax1 = plt.subplots(figsize=(6, 6))
ax1.set_title("Training losses on different experiments for dataset 100")
ax1.set_xlabel("Epoch[-]")
ax1.set_ylabel("Loss[-]")
ax1.set_ylim([0,1.5])
for v in loss:
    x=v['data'][0]['x']
    y=v['data'][0]['y']
    ax1.plot(x,y)
ax1.legend(experiment)
plt.savefig(log_dir +'train_loss_comarision', bbox_inches='tight', pad_inches=0.1)

fig, ax2 = plt.subplots(figsize=(6, 6))
ax2.set_title("Validation losses on different experiments for dataset 100")
ax2.set_xlabel("Epoch[-]")
ax2.set_ylabel("Loss[-]")
ax2.set_ylim([0,1.5])
for v in loss:
    x=v['data'][1]['x']
    y=v['data'][1]['y']
    ax2.plot(x,y)
ax2.legend(experiment)
plt.savefig(log_dir +'valid_loss_comarision', bbox_inches='tight', pad_inches=0.1)

fig, ax1 = plt.subplots(figsize=(6, 6))
ax1.set_title("Training Top-1 accurracy on different experiments for dataset 100")
ax1.set_xlabel("Epoch[-]")
ax1.set_ylabel("Top-1 accurracy[-]")
#ax1.set_ylim([0,1])
for v in acc:
    x=v['data'][0]['x']
    y=v['data'][0]['y']
    ax1.plot(x,y)
ax1.legend(experiment)
plt.savefig(log_dir +'train_acc_comarision', bbox_inches='tight', pad_inches=0.1)

fig, ax2 = plt.subplots(figsize=(6, 6))
ax2.set_title("Validation Top-1 accurracy on different experiments for dataset 100")
ax2.set_xlabel("Epoch[-]")
ax2.set_ylabel("Top-1 accurracy[-]")
#ax2.set_ylim([0,1])
for v in acc:
    x=v['data'][1]['x']
    y=v['data'][1]['y']
    ax2.plot(x,y)
ax2.legend(experiment)
plt.savefig(log_dir +'valid_acc_comarision', bbox_inches='tight', pad_inches=0.1)

print("Test Top-1 accuracy on different experiments for dataset 100")
print("Version\tAccuracy")
for v in results:
    print('{0}\t\t{1:.2f}%'.format(experiment[results.index(v)],v['test_acc']*100.))
with open(log_dir + 'results.txt','w') as f:
    print("Test Top-1 accuracy on different experiments for dataset 100",file=f)
    print("Version\tAccuracy",file=f)
    for v in results:
        print('{0}\t\t{1:.2f}%'.format(experiment[results.index(v)],v['test_acc']*100.),file=f)

