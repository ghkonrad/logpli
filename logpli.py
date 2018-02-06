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
import sys

import argparse

import math
import numpy
import numpy.linalg

import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate
import scipy.optimize 
import scipy.ndimage

import csv
import collections

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Wygładzanie algorytmem opartym o LZNK.');
parser.add_argument('plik', type=str, help='Plik z danymi')
parser.add_argument('N', type=int, nargs='?', default=0, help='Numer kolumny, liczony od zera')
parser.add_argument('--delimiter', type=str, default='\t', help='Znak służący do oddzielania kolumn w pliku wejściowym CSV')
parser.add_argument('--title', type=str, default="", help='Ustawienie określonego tytułu w obrazkach')
parser.add_argument('--savecsv', type=str, default=None, help='Zapisanie danych do podanego pliku')
parser.add_argument('--n0', type=int, default=1, help='Początkowy zakres uśredniania')
parser.add_argument('--n1', type=int, default=100, help='Końcowy zakres uśredniania')
parser.add_argument('--iters', type=int, default=1, help='Pozwala wykonać kilka iteracji')
parser.add_argument('--Jodj', type=float, default=None, help='Stała odejmowana od J eksperymentalnego')
parser.add_argument('--tmin', type=float, default=-numpy.inf, help='Minimalne t')
parser.add_argument('--tmax', type=float, default=numpy.inf, help='Maksymalne t')
parser.add_argument('--tpoint', type=float, default=None, help='Orientacyjny punkt na wykresach podany względem czasu')
parser.add_argument('--st', type=int, default=1, help='Stopień wielomianu w LZNK')
parser.add_argument('--nielinznk', action='store_true', default=False, help='Stosuj nieliniowe zadanie najmniejszych kwadratów (domyślnie: LZNK)')
parser.add_argument('--gaussian', action='store_true', default=False, help='Stosuj filtrowanie Gaussa na danych wejściowych (domyślnie: nie)')
parser.add_argument('--najpierwusredn', action='store_true', default=False, help='Uśrednij przed odjęciem Jodj (domyślnie: uśredniamy po odjęciu)')
args = parser.parse_args()


def deriv(v, t):
	assert(len(v)>1);
	assert(len(v) == len(t));
	dv = [(v[1] - v[0])/(t[1]-t[0])];
	for i in range(1,len(v)-1):
		dv.append( (v[i+1] - v[i])/(t[i+1] - t[i]) );
		#dv.append( (v[i+1] - v[i-1])/(t[i+1] - t[i-1]) );
	dv.append((v[-1] - v[-2])/(t[-1]-t[-2]));
	return numpy.array(dv);

def exp_plus_c_fit(x, fx, c0):
	def fun(c, x, y):
		return (numpy.exp(c[0] * x + c[1]) + c[2]) - y;

	res = scipy.optimize.least_squares(fun, c0, args=(x,fx))
	c = res.x;
	#print(c0, c);
	fit = fun(c,x,0);
	return (fit,c);

def load_exp_data(plik, kol_J = 1, delimiter=' ', **kwords):
	t_J = numpy.genfromtxt(plik, delimiter=delimiter, **kwords)

	t = t_J[:,0];# * un.ns;
	J = t_J[:,kol_J];

	return (t,J);

def lznk_fit(x, fx, st):
	A = numpy.vstack([x**i for i in range(st+1)]).T;
	c = numpy.linalg.lstsq(A,fx)[0];
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
			dJt = deriv(Jt,tt);
			dlnJt = dJt/Jt;
		else:
			lnJt,_ = lznk_fit(tt, numpy.log(Jg[zakr]), st=st);
			dlnJt = deriv(lnJt,tt);
			Jt = numpy.exp(lnJt);
			dJt = deriv(Jt,tt);
		#lnJt,_ = lznk_fit(tt, numpy.log(Jg[zakr]), st=st);
		#dlnJt = deriv(lnJt,tt);
		#Jt = numpy.exp(lnJt);
		#dJt = deriv(Jt,tt);
		
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


def main():

	columns = read_data_headers(args.plik, delimiter=args.delimiter)

	if(args.N == 0):

		print("Podaj kolumnę do uśrednienia jako ostatni parametr.\nMożliwości:");
		for col in sorted(columns.values(), key= lambda col: col.number):
			print("%3d: %3s (%s)"%((col.number, col.name, col.label)));

	else:
		if(args.N not in columns):
			print("Nie odnalazłem kolumny {args.N}. Może jest źle ustawiony parametr --delimiter?".format(args=args));
			sys.exit(2);

		col=columns[args.N];
		print("Opracujemy kolumnę %d: %s (%s)"%((col.number, col.name, col.label)));

		title = "%s --- %s (%s)"%((args.title, col.name, col.label));

		[te, Je] = load_exp_data(args.plik, args.N, delimiter=args.delimiter);

		indices = numpy.where(numpy.logical_and(te >= args.tmin, te <= args.tmax));
		te = te[indices];
		Je = Je[indices];

		Jo = numpy.copy(Je);

		if(args.Jodj is not None):
			Jodj = args.Jodj;
		else:
			nfit = args.n1;
			(Jodjfit,c) = exp_plus_c_fit(te[-nfit:], Je[-nfit:], [0,0,0]);
			Jodj = c[2];
			print("Znalezione Jodj: %f"%Jodj);

			plt.plot(te, Je, color = 'green', label="$J$ experimental");
			plt.plot(te[-nfit:], Jodjfit, color='blue', label="$J$ fit");
			plt.yscale('log');
			plt.legend(loc='best');
			plt.xlabel("$t$");
			plt.title(title);
			plt.show();

		Je = numpy.abs(Je - Jodj);
		dJe = deriv(Je, te);
		dlnJe = dJe/Je;


		for iter in range(args.iters):
			if(iter==0):
				tin = te;
				if(args.najpierwusredn):
					Jin = Jo;
				else:
					Jin = Je;
			else:
				tin = ts;
				Jin = Jts;


			(ts, Jts, dlnJts, _) = new_approx(tin, Jin, args.n0, args.n1, st=args.st, nielinznk=args.nielinznk, gaussian=args.gaussian);

		if(args.najpierwusredn):
			Jts -= Jodj;

		if(args.tpoint is not None):
			point = (numpy.abs(ts-args.tpoint)).argmin();

		plt.plot(te, Je, color = 'green', label="$J$ experimental");
		plt.plot(ts, Jts, color='red', label="$J$ denoised");
		if(args.tpoint is not None):
			plt.plot(ts[point], Jts[point], 'ro');
		plt.yscale('log');
		plt.legend(loc='best');
		plt.xlabel("$t$");
		plt.title(title);
		plt.show();

		#dJe = deriv(Je, te);
		#plt.plot(te, dJe, color='green', label=r"$\frac{d J}{d t}$ experimental");
		#plt.plot(ts, dlnJts*Jts, color='red', label=r"$\frac{d J}{d t}$ denoised");
		#if(args.tpoint is not None):
		#	plt.plot(ts[point], dlnJts[point]*Jts[point], 'ro');
		#plt.yscale('linear');
		#plt.legend(loc='best');
		#plt.xlabel("$t$");
		#plt.title(title);
		#plt.show();
		
		plt.plot(Je , -dlnJe , color='green', label=r"$-\frac{d \log(J)}{d t}$ experimental");
		plt.plot(Jts, -dlnJts, color='red',   label=r"$-\frac{d \log(J)}{d t}$ denoised");
		if(args.tpoint is not None):
			plt.plot(Jts[point], -dlnJts[point], 'ro');
		plt.yscale('linear');
		plt.ylim(0.5*min(-dlnJts), 1.15*max(-dlnJts));
		plt.legend(loc='best');
		plt.xlabel("$J$");
		plt.title(title);
		plt.show();
		
		plt.plot(numpy.sqrt(Je),  -dlnJe , color='green', label=r"$-\frac{d \log(J)}{d t}$ experimental");
		plt.plot(numpy.sqrt(Jts), -dlnJts, color='red',   label=r"$-\frac{d \log(J)}{d t}$ denoised");
		if(args.tpoint is not None):
			plt.plot(numpy.sqrt(Jts[point]), -dlnJts[point], 'ro');
		plt.yscale('linear');
		plt.ylim(0.5*min(-dlnJts), 1.15*max(-dlnJts));
		plt.legend(loc='best');
		plt.xlabel(r"$\sqrt{J}$");
		plt.title(title);
		plt.show();


		if(args.savecsv is not None):
			numpy.savetxt(args.savecsv, numpy.column_stack((te, Jo, Je, -dlnJe, ts, Jts, -dlnJts)),
				fmt='%15.7e',
				delimiter='\t',
				header='\t'.join(('%15s'%s for s in ('te', 'Jo', 'Je', '-dlnJe', 'ts', 'Jts', '-dlnJts'))),
				comments='',
				);



main();
