import matplotlib.pyplot as plt
import seaborn as sns
import torch
sns.set(rc={'figure.figsize':(8, 8)})
def plot_mmd_vs(all_mmd_vs, index=0, all=False, alpha=1e5, save=None, _type='mmd'):
	'''
	Args: 
	all_mmd_vs: tensor of shape n_participants * n_participants *  M_samples
	index: 0 == joint, 1 == 1st participant, and so on
	Return:
	plot the mmd comparison density plot for index
	'''

	mmds_cumulative = []
	for i, mmds in enumerate(all_mmd_vs[index]):

		if i == index:
			label = 'vs self'
			sns.kdeplot(mmds * alpha, label=label)
		else:
			if i == 0:
				label = 'vs joint'
			else:
				label = 'vs {}'.format(i)

			mmds_cumulative.append(mmds)

			if all:
				sns.kdeplot(mmds * alpha, label=label)

		# plt.hist(mmds, label=label, density=True)
	sns.kdeplot(torch.cat(mmds_cumulative) * alpha, label='vs others')

	if index == 0:
		title = 'Joint vs all {} densities'.format(_type)
	else:
		title = str(index) + ' vs all {} densities'.format(_type)
	plt.title(title)
	
	plt.xlabel('{} values'.format(_type))
	# Set the y axis label of the current axis.
	plt.ylabel('density')
	# Set a title of the current axes.
	# show a legend on the plot
	plt.legend()
	# Display a figure.

	if save:
		plt.savefig(save)
	else:
		plt.show()
	plt.clf()
