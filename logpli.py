#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################################################
#   Copyright 2018 Konrad Sakowski, Stanislaw Krukowski, Pawel Strak
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#############################################################################
#

import argparse
import collections
import csv
from functools import cached_property
from IPython.display import clear_output, display, HTML, Markdown, Latex
import matplotlib.pyplot as plt

import numpy
import numpy.linalg

import pint

import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate
import scipy.optimize 
import scipy.ndimage

import sys


un_default = pint.UnitRegistry(system='mks');

class EstimateABC(object):
	"""
		Estimate ABC constants.
		
		Prameters:
		* *fit_deg* --- degree of fitting polynomial, at least 2
		* *un* --- a :class:`pint.UnitRegistry` instance
	"""
		#* *time_unit* --- time unit on plots and in the text file to/from *logpli*
		#* *fit_deg* --- degree of fitting polynomial, at least 2
		#* *fig_prefix* --- if not None, figures will be saved in this directory
		#* *mkdir* --- if the *fig_prefix* directory shall be created if it does not exist
	#def __init__(self, time_unit, fit_deg = 2, fig_prefix=None, mkdir=False, un = un_default):
	def __init__(self, fit_deg = 2, un = un_default):
		assert(fit_deg >= 2);

		self.fit_deg = fit_deg;

		self.un = un;
		

	def limit_args(self, x, y, x_min, x_max):
		"""
			Limit argument vector *x* and value vector *y* so that only arguments in [*x_min*, *x_max*] remain.

			Return: (x,y) with arguments in [*x_min*, *x_max*].
		"""
		indices = numpy.where(numpy.logical_and(x >= x_min, x <= x_max));

		return (x[indices], y[indices]);

	@cached_property
	def plot(self):
		"""
			Return a compatible :class:`Plotter` instance to perform plots.
			Multiple uses results in the same instance being returned.
		"""
		return Plotter(un=self.un);

	def remove_nonzero(self, x, y):
		"""
			Remove arguments and values for indices where the value is 0.

			Parameters:

			* *x* --- argument vector
			* *y* --- value vector

			Return: (*x*, *y*) with zero values removed
		"""
		assert x.shape == y.shape

		nonzero_idcs = y.magnitude.nonzero();
		return (x[nonzero_idcs], y[nonzero_idcs]);

	def smooth(self, te, Je, Jodj=None, **smoothing_kwargs):
		"""
			Smooth experimental data.

			Parameters:

			* *te* --- time (with units)
			* *Je* --- (noisy) data to be de-noised (with units)
			* *Jodj* --- value to be subtracted from Je (with units)
			* All optional parameters of :func:`smoothing` may be passed.

			Return: a named tuple (*ts*, *Jts*, *mdlnJts*) 

			* *ts*  --- a vector of time steps for smoothed luminescence intensity
			* *Js* --- a vector of smoothed luminescence intensity
			* *mdlnJs*  --- a vector of negative logarithmic derivative of luminescence intensity
		"""

		Return = collections.namedtuple("Return", ["ts", "Js", "mdlnJs"])
		#plt.plot(te , Je, **self.style_data);
		#plt.yscale("log");
		#plt.xlabel(f"Time [${self.time_unit:~L}$]");
		#plt.ylabel(f"Optical output [arb. u.]");
		#self._save_fig("Je");
		#plt.show();

		if(Jodj != None):
			Jodj = (0*Je.units + Jodj); # this is to ensure Jodj unit is compatible with Je unit

		ret = smoothing(
			te = te.magnitude,
			Je = Je.magnitude,
			Jodj = Jodj.to(Je.units).magnitude,
			**smoothing_kwargs
			);

		return Return(
			ts = ret.ts * te.units,
			Js = ret.Jts * Je.units, 
			mdlnJs = ret.mdlnJts / te.units); 

class Plotter(object):
	"""
		A class for some standard auxiliary plots.
	"""
	def __init__(self, un=un_default):
		self.style_data = {'color':'#1F20E0', 'label':'orig.'};
		self.style_smoothed = {'color':'#E07E1F', 'label':'smooth.'};
		self.style_fit = {'color':'#1FE07E'};
		self.style_fit_outside_interval = {'color':'#1FE07E', "ls":"--"}; # approximation outside of fitting interval

		self.un=un;

	def Je(self, te, Je):
		"""
			Plot of experimentally measured luminescence output *Je* versus time *te*.
		"""
		plt.plot(te.magnitude, Je.magnitude, **self.style_data);
		plt.yscale("log");
		plt.xlabel(f"Time [${te.units:~L}$]");
		plt.ylabel(f"Optical output [arb. u.]");
		plt.legend();
		plt.show();

	def JeJs(self, te, Je, ts, Js):
		"""
			Plot of experimentally measured luminescence output *Je* versus time *te* and smoothed luminescence output *Js* versus time *ts*
		"""
		assert te.units == ts.units
		assert Je.units == Js.units

		plt.plot(te.magnitude, Je.magnitude, **self.style_data);
		plt.plot(ts.magnitude, Js.magnitude, **self.style_smoothed);
		plt.yscale("log");
		plt.xlabel(f"Time [${te.units:~L}$]");
		plt.ylabel(f"Optical output [arb. u.]");
		plt.legend();
		plt.show();

	def rel_Je_vs_Js(self, te, Je, ts, Js):
		"""
			Plot of relative difference of experimentally measured luminescence output *Je* (time *te*) and smoothed luminescence output *Js* (time *ts*).
		"""
		assert te.units == ts.units
		assert Je.units == Js.units

		time_unit = te.units;

		te = te.magnitude;
		Je = Je.magnitude;
		ts = ts.magnitude;
		Js = Js.magnitude;

		f_Js = scipy.interpolate.interp1d(ts, Js, kind="linear");

		plt.plot(te, (Je - f_Js(te)) / f_Js(te), label="(orig. - smooth.) / smooth.", color=self.style_data["color"]);
		plt.plot(te, 0*te, **self.style_smoothed);
		plt.yscale("linear");
		plt.xlabel(f"Time [${time_unit:~L}$]");
		plt.ylabel(f"Rel. difference");
		plt.legend();
		plt.show();

def exp_plus_c_fit(x, fx, c0):
	def fun(c, x, y):
		return (numpy.exp(c[0] * x + c[1]) + c[2]) - y;

	res = scipy.optimize.least_squares(fun, c0, args=(x,fx))
	c = res.x;
	#print(c0, c);
	fit = fun(c,x,0);
	return (fit,c);

def load_exp_data(plik, kol_t=0, kol_J = 1, delimiter=' ', **kwords):
	t_J = numpy.genfromtxt(plik, delimiter=delimiter, **kwords)

	t = t_J[:,kol_t];# * un.ns;
	J = t_J[:,kol_J];

	return (t,J);

def lznk_fit(x, fx, st):
	A = numpy.vstack([x**i for i in range(st+1)]).T;
	c = numpy.linalg.lstsq(A,fx,rcond=None)[0];
	#print(c);
	fit = sum(c[i] * x**i for i in range(st+1));
	return (fit,c);

def lznk_exp_fit(x, fx, c0, st):
	def fun(c, x, y):
		return numpy.exp(sum([c[i] * (x**i) for i in range(st+1)])) - y;

	res = scipy.optimize.least_squares(fun, c0, args=(x,fx))
	c = res.x;
	#print(c0, c);
	fit = numpy.exp(sum([c[i] * (x**i) for i in range(st+1)]));
	return (fit,c);

def new_approx(te,Je,initr, endr, st, nielinznk=False, gaussian=False):

	assert(initr>=1);
	assert(initr<=endr);

	if(endr >= len(Je)):
		raise RuntimeError("Final averaging interval is larger than total data length");

	if(gaussian):
		n_gaussian = 5
		Jg = scipy.ndimage.filters.gaussian_filter1d(Je, n_gaussian, mode='nearest');
		Jg[:n_gaussian] = Je[:n_gaussian];
		Jg[-n_gaussian:] = Je[-n_gaussian:];
	else:
		Jg = Je

	dlnJts = [];
	dJts = [];
	Jts = [];
	ts = [];
	for i in range(len(Jg)):
		#prom =  initr + endr * (i - initr) / (len(Jg) - initr - endr);
		fac = min(1, float(i) / (len(Jg) - endr))
		prom =  int(initr*(1-fac) + endr*fac);

		zakr, poz = zakres(i, prom, len(Jg)); 
		#if(i<=initr):
		#	#print(1);
		#	zakr = range(2*initr+1);
		#	poz = i;
		#elif(i > len(Jg)-initr-endr):
		#	#print(2);
		#	zakr = range(len(Jg)-2*(initr+endr) + 1,len(Jg));
		#	poz = i + 2*initr + 2*endr - len(Jg) -1;
		#else:
		#	#print(3);
		#	zakr = range(i-prom, i+prom+1);
		#	poz = len(zakr)/2;

		#print(zakr);
		##print(i,poz, zakr[poz]);

		assert(i == zakr[poz]);

		tt = te[zakr];
		#print(te[i], tt[poz]);

		if(nielinznk):
			_,c0 = lznk_fit(tt, numpy.log(Je[zakr]), st=st);
			Jt,_ = lznk_exp_fit(tt, Jg[zakr], c0=c0, st=st);
			dJt = numpy.gradient(Jt,tt);
			dlnJt = dJt/Jt;
		else:
			lnJt,_ = lznk_fit(tt, numpy.log(Jg[zakr]), st=st);
			dlnJt = numpy.gradient(lnJt,tt);
			Jt = numpy.exp(lnJt);
			dJt = numpy.gradient(Jt,tt);
		
		ts.append(te[i]);
		Jts.append(Jt[poz]);
		dlnJts.append(dlnJt[poz]);
		if(i<len(Jg)-1):
			dJts.append((Jt[poz+1]-Jt[poz])/(tt[poz+1]-tt[poz]));
		else:
			dJts.append(dJts[-1]);

	dlnJts = numpy.array(dlnJts);
	dJts = numpy.array(dJts);
	Jts = numpy.array(Jts);
	ts = numpy.array(ts);

	return (ts, Jts, dlnJts, Jt);

def read_data_headers(plik, delimiter=' ', **kwords):
	with open(plik, 'r') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=delimiter, quotechar='"')
		linia=0;
		for row in csvreader:
			#print(row)
			linia += 1;
			if(linia == 1):
				names = row;
			elif(linia == 2):
				labels = row;
			elif(linia == 3):
				break;

		assert(len(names) == len(labels));


	Column = collections.namedtuple('Column', ['number', 'name', 'label']);

	columns = dict();

	for number, (name, label) in enumerate(zip(names, labels)):
		columns[number] = Column(number = number, name = name, label = label);

	return columns;

# Zwraca zakres indeksów zawarty w [0,N), który jest długości 2*n+1 i zwykle środkiem jest liczba "i", poza sytuacją, gdy "i" jest zbyt blisko brzegów i po prostu zwracany jest zakres przy brzegu o długości średnicy
def zakres(i,n, N):
	srednica = 2*n + 1;
	if(N <= srednica):
		zakr = range(N);
		poz = i;
	else:
		if(i<n):
			zakr = range(srednica);
			poz = i;
		elif(i>=N-n):
			zakr = range(N-srednica, N);
			poz = 2*n + 1 -(N-i);
		else:
			zakr = range(i-n, i+n+1);
			poz = n;

	return (zakr, poz);


def smoothing(
	te, Je,
	st=1,
	n0=1,
	n1=100,
	iters=1,
	Jodj=None,
	nielinznk=False,
	gaussian=False,
	averageoutfirst=False,
	verbose=False,
	plots=False,
	title = "",
	tpoint=None,
	):
	"""
		Smoothing of noisy numerical data.

		Parameters:

		* *te* --- a vector of (increasing) experimental time steps
		* *Je* --- a vector of corresponding experimental luminescence intensity
		* *st* --- polynomial degree for least-squares fitting
		* *n0* --- initial averaging interval
		* *n1* --- final averaging interval
		* *iters* --- multiple iterations of the averaging
		* *Jodj* --- constant subtracted from the experimental J data (default: *None*, find automatically)
		* *nielinznk* --- use nonlinear least squares (default is to use linear)
		* *gaussian* --- use Gaussian filtering (default: no)
		* *averageoutfirst* --- first average out J, then subtract Jodj (default is to subtract Jodj from J and then to average out)
		* *verbose* --- show text
		* *plots* --- show plots
		* *title* --- title of produced figures
		* *tpoint* --- auxiliary indicatory point to be put on the figures (default: *None*, no point)

		Return: a named tuple of:

		* *ts*  --- a vector of time steps for smoothed luminescence intensity
		* *Jts* --- a vector of smoothed luminescence intensity
		* *mdlnJts*  --- a vector of negative logarithmic derivative of luminescence intensity
		* *te* --- a vector of experimental time steps
		* *Je* --- a vector of corresponding experimental luminescence intensity with *Jodj* subtracted
		* *Je* --- a vector of corresponding original experimental luminescence intensity
		* *mdlnJe*  --- a vector of negative logarithmic derivative of luminescence intensity, computed directly from the experimental data


	"""

	SmoothingReturn = collections.namedtuple("SmoothingReturn", ["ts", "Jts", "mdlnJts", "te", "Je", "mdlnJe", "Jo"])

	# we will work on copies
	te = numpy.copy(te);
	Je = numpy.copy(Je);

	# original data, stored separately
	Jo = numpy.copy(Je);

	if(Jodj is None):
		nfit = n1;
		(Jodjfit,c) = exp_plus_c_fit(te[-nfit:], Je[-nfit:], [0,0,0]);
		Jodj = c[2];
		if(verbose):
			print("Jodj found: %f"%Jodj);

		if(plots):
			plt.plot(te, Je, color = 'green', label="$J$ experimental");
			plt.plot(te[-nfit:], Jodjfit, color='blue', label="$J$ fit");
			plt.yscale('log');
			plt.legend(loc='best');
			plt.xlabel("$t$");
			plt.title(title);
			plt.show();

	Je = numpy.abs(Je - Jodj);
	dJe = numpy.gradient(Je, te);
	dlnJe = dJe/Je;


	for iter in range(iters):
		if(iter==0):
			tin = te;
			if(averageoutfirst):
				Jin = Jo;
			else:
				Jin = Je;
		else:
			tin = ts;
			Jin = Jts;


		(ts, Jts, dlnJts, _) = new_approx(tin, Jin, n0, n1, st=st, nielinznk=nielinznk, gaussian=gaussian);

	if(averageoutfirst):
		Jts -= Jodj;

	if(tpoint is not None):
		point = (numpy.abs(ts-tpoint)).argmin();

	if(plots):
		plt.plot(te, Je, color = 'green', label="$J$ experimental");
		plt.plot(ts, Jts, color='red', label="$J$ denoised");
		if(tpoint is not None):
			plt.plot(ts[point], Jts[point], 'ro');
		plt.yscale('log');
		plt.legend(loc='best');
		plt.xlabel("$t$");
		plt.title(title);
		plt.show();

		#dJe = numpy.gradient(Je, te);
		#plt.plot(te, dJe, color='green', label=r"$\frac{d J}{d t}$ experimental");
		#plt.plot(ts, dlnJts*Jts, color='red', label=r"$\frac{d J}{d t}$ denoised");
		#if(tpoint is not None):
		#	plt.plot(ts[point], dlnJts[point]*Jts[point], 'ro');
		#plt.yscale('linear');
		#plt.legend(loc='best');
		#plt.xlabel("$t$");
		#plt.title(title);
		#plt.show();

		plt.plot(Je , -dlnJe , color='green', label=r"$-\frac{d \log(J)}{d t}$ experimental");
		plt.plot(Jts, -dlnJts, color='red',   label=r"$-\frac{d \log(J)}{d t}$ denoised");
		if(tpoint is not None):
			plt.plot(Jts[point], -dlnJts[point], 'ro');
		plt.yscale('linear');
		plt.ylim(0.5*min(-dlnJts), 1.15*max(-dlnJts));
		plt.legend(loc='best');
		plt.xlabel("$J$");
		plt.title(title);
		plt.show();

		plt.plot(numpy.sqrt(Je),  -dlnJe , color='green', label=r"$-\frac{d \log(J)}{d t}$ experimental");
		plt.plot(numpy.sqrt(Jts), -dlnJts, color='red',   label=r"$-\frac{d \log(J)}{d t}$ denoised");
		if(tpoint is not None):
			plt.plot(numpy.sqrt(Jts[point]), -dlnJts[point], 'ro');
		plt.yscale('linear');
		plt.ylim(0.5*min(-dlnJts), 1.15*max(-dlnJts));
		plt.legend(loc='best');
		plt.xlabel(r"$\sqrt{J}$");
		plt.title(title);
		plt.show();

	return SmoothingReturn(
		ts = ts,
		Jts = Jts,
		mdlnJts = -dlnJts,
		te = te,
		Je = Je,
		mdlnJe = -dlnJe,
		Jo = Jo,
		);


def main():

	parser = argparse.ArgumentParser(description='(Linear-)least-squares-based fitting algorithm.');
	parser.add_argument('sourcefile', type=str, help='CSV data file')
	parser.add_argument('N', type=int, nargs='?', default=0, help='Column number, indexed from 0')
	parser.add_argument('--ct', type=int, default=0, help='Column number for time (default: 0)')
	parser.add_argument('--delimiter', type=str, default='\t', help='CSV column delimiter')
	parser.add_argument('--title', type=str, default="", help='Title of produced figures')
	parser.add_argument('--savecsv', type=str, default=None, help='Save results to this file')
	parser.add_argument('--n0', type=int, default=1, help='Initial averaging interval')
	parser.add_argument('--n1', type=int, default=100, help='Final averaging interval')
	parser.add_argument('--iters', type=int, default=1, help='Multiple iterations of the averaging')
	parser.add_argument('--Jodj', type=float, default=None, help='Constant subtracted from the experimental J data')
	parser.add_argument('--tmin', type=float, default=-numpy.inf, help='Minimal t')
	parser.add_argument('--tmax', type=float, default=numpy.inf, help='Maximal t')
	parser.add_argument('--tpoint', type=float, default=None, help='Auxiliary indicatory point to be put on the figures')
	parser.add_argument('--st', type=int, default=1, help='Polynomial degree for least-squares fitting')
	parser.add_argument('--nielinznk', action='store_true', default=False, help='Use nonlinear least squares (default is to use linear)')
	parser.add_argument('--gaussian', action='store_true', default=False, help='Use Gaussian filtering (default: no)')
	parser.add_argument('--averageoutfirst', action='store_true', default=False, help='First average out J, then subtract Jodj (default is to subtract Jodj from J and then to average out)')
	parser.add_argument('--noplots', action='store_true', default=False, help='Suppress plots')
	parser.add_argument('--ignorezeros', action='store_true', default=False, help='Ignore measurements which were equal to 0')
	args = parser.parse_args()

	plots = not args.noplots;

	columns = read_data_headers(args.sourcefile, delimiter=args.delimiter)

	if(args.N == 0):

		print("Specify the column to average out as a last parameter.\nList of columns:");
		for col in sorted(columns.values(), key= lambda col: col.number):
			print("%3d: %3s (%s)"%((col.number, col.name, col.label)));

		if(len(columns)==1):
			print("Only one column in this file? Perhaps --delimiter is not set correctly?".format(args=args));

	else:
		if(args.N not in columns):
			print("No column {args.N}. Perhaps --delimiter is not set correctly?".format(args=args));
			sys.exit(2);

		colx = columns[args.ct];
		col = columns[args.N];
		print("Using columns:");
		print("X: %d, %s (%s)"%((colx.number, colx.name, colx.label)));
		print("Y: %d, %s (%s)"%((col.number, col.name, col.label)));

		title = "%s --- %s (%s)"%((args.title, col.name, col.label));

		[te, Je] = load_exp_data(plik=args.sourcefile, kol_t=args.ct, kol_J=args.N, delimiter=args.delimiter, skip_header=1);

		if(args.ignorezeros):
			# remove zeros
			nonzero_idcs = Je.nonzero();
			te = te[nonzero_idcs];
			Je = Je[nonzero_idcs];


		indices = numpy.where(numpy.logical_and(te >= args.tmin, te <= args.tmax));
		te = te[indices];
		Je = Je[indices];



		ret = smoothing(
			te, Je,
			st = args.st,
			n0 = args.n0,
			n1 = args.n1,
			iters = args.iters,
			Jodj = args.Jodj,
			nielinznk = args.nielinznk,
			gaussian = args.gaussian,
			averageoutfirst = args.averageoutfirst,
			verbose = True,
			plots = plots,
			title = title,
			tpoint = args.tpoint,
			);

		if(args.savecsv is not None):
			numpy.savetxt(args.savecsv, numpy.column_stack((ret.te, ret.Jo, ret.Je, ret.mdlnJe, ret.ts, ret.Jts, ret.mdlnJts)),
				fmt='%15.7e',
				delimiter=args.delimiter,
				header=args.delimiter.join(('%15s'%s for s in ('te', 'Jo', 'Je', '-dlnJe', 'ts', 'Jts', '-dlnJts'))),
				comments='',
				);

if __name__ == "__main__":
	main();
