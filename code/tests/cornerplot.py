"""
Simple library to create corner plots in `matplotlib`.

Corner plots provide a simple way to visualize multidimensional data,
with each dimension plotted against every other dimension.
"""

__author__    = 'Mathieu Daëron'
__contact__   = 'daeron@lsce.ipsl.fr'
__copyright__ = 'Copyright (c) 2023 Mathieu Daëron'
__license__   = 'MIT License - https://opensource.org/licenses/MIT'
__date__      = '2023-10-05'
__version__   = '1.0'


import matplotlib.pyplot as _ppl


class Cornerplots():

	def __init__(self,
		fields,
		labels = None,
		fig = None,
		subplots_adjust = None,
		sharexy = True,
		internal_ticklabels = False,
		grid_kwargs = None,
		):
		"""
		Create triangle of axes plotting each field against each other once
	
		Parameters:
			fields (list): list of strings to be used as data fields
			labels (list): list of strings to be used as axis labels (default: use fields)
			fig (matplotlib.Figure): exisiting figure (default: create new figure)
			subplots_adjust (tuple): arguments to pass to `subplots_adjust()`
			sharexy (bool): force all X- and Y-axes of the same field to have identical limits (default: True)
			internal_ticklabels (bool): whether to show ticklabels on every subplot (default: False)
			grid_kwargs (dict): kwargs to be passed to Axes.grid() (default: no grid() call)
		"""

		self.fields = [_ for _ in fields]
		self.labels = [_ for _ in fields] if labels is None else [_ for _ in labels]
		self.fig = _ppl.gcf() if fig is None else fig
		self.grid_kwargs = grid_kwargs
		if subplots_adjust is None:
			if internal_ticklabels:
				self.subplots_adjust = (.12, .12, .95, .95, .3, .3)
			else:
				self.subplots_adjust = (.12, .12, .95, .95, .15, .15)
		else:
			self.subplots_adjust = subplots_adjust

		_ppl.figure(self.fig)
		_ppl.subplots_adjust(*self.subplots_adjust)

		N = len(labels)
		axes = {}

		for j in range(N):
			for k in range(0, j+1):

				ax = _ppl.subplot(
					N, N, N*j + k + 1,
					sharex = axes[(k, j-1)] if sharexy and j > 0 and k < j else None,
					sharey = axes[(0, j)] if sharexy and k > 0 else None,
					)

				if j == (N-1):
					_ppl.xlabel(labels[k])
				else:
					if not internal_ticklabels:
						_ppl.setp(ax.get_xticklabels(), visible = False)
						ax.tick_params(axis = 'x', length = 0)
				if k == 0:
					_ppl.ylabel(labels[j])
				else:
					if not internal_ticklabels:
						_ppl.setp(ax.get_yticklabels(), visible = False)
						ax.tick_params(axis = 'y', length = 0)

				if self.grid_kwargs:
					ax.grid(**self.grid_kwargs)

				axes[(k, j)] = ax

		self.axes = {(self.fields[i], self.fields[j]): axes[(i, j)] for i, j in axes}
		self.legend_items = []


	def plot(self, datadict, ydata=None, *args, **kwargs):
		"""
		Plot data in the proper location of a cornerplot
	
		Parameters:
			datadict (dict): a dictionary of the form `{f1: <array-like>, f2: <array-like>, ...}`,
				where `f1`, `f2`, ... are fields corresponding to each array-like to plot.
				These fields will determine which subplot to use, and whether each field
				should plot along the X or Y axis.
			ydata (dict): optional dictionary for y-axis data. If provided, datadict is used for x-axis
				and ydata is used for y-axis. Useful for plotting e.g. true vs predicted values.
			*args, **kwargs: to be passed on to `pyplot.plot()`
	
		Returns:
			List of values returned by `pyplot.plot()`
		"""
		# If ydata not provided, use datadict for both axes (original behavior)
		if ydata is None:
			ydata = datadict
			
		indices = sorted([self.fields.index(_) for _ in datadict])
		fields = [self.fields[_] for _ in indices]

		result = []
		for j, fj in enumerate(fields):
			for i, fi in enumerate(fields[:j+1]):
				result += self.axes[(fi, fj)].plot(datadict[fi], ydata[fj], *args, **kwargs)

		if not result[0].get_label().startswith('_'):
			self.legend_items.append(result[0])

		return result


	def legend(self, *args, **kwargs):
		"""
		Plot combined legend for all cornerplots
	
		Parameters:
			*args, **kwargs: to be passed on to `pyplot.legend()`
	
		Returns:
			Value returned by `pyplot.legend()`
		"""
		return _ppl.legend(
			self.legend_items,
			[_.get_label() for _ in self.legend_items],
			*args,
			**(kwargs | {
				'loc': 'upper right',
				'bbox_to_anchor': self.subplots_adjust[2:4],
				'bbox_transform': self.fig.transFigure,
				'borderaxespad': 0,
				}),
			)