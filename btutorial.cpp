/*
	3 Layer Neural Network Simulator ( Tutorial Version ) with Boost
	Training Algorithm : Back Propagation ( Jump Every Time Version )

	This program is a 'Boost version' of tutorial.c.
	Results of execution of this program is comfirmed
	to be identical as that of tutorial.c because procedure are
	identical to tutorial.c.

	Boost library is necessary to compile this source.
	Please refer to the Boost web site (http://www.boost.org/)
	for more detailed information.

	$HeadURL$
*/

static char id[]="$Id$";

// To accelate calculations
#define BOOST_UBLAS_NDEBUG

//#include <stdio.h>
#include <stdlib.h>
// for use of exit(), drand48(), srand48()
#include <math.h>
// for use of tanh()

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS	0
#endif
#ifndef EXIT_FAILURE
#define EXIT_FAILURE	1
#endif

#include <iostream>
#include <fstream>
//#include <complex>

// Boost headers
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>

using namespace boost::numeric::ublas;

#define INP 3					/* number of input units  */
#define HID 2					/*           hidden units */
#define OUT 1					/*           output units */

#define EX_NUM 8           		/* number of input patterns */
#define MAX_ITER	10000		/* maximum of iteration */
#define MIN_ERROR	0.001		/* destination of error rms */

#define IWR			1.0			/* amplitude of initial weight */
#define ETA			0.1			/* training rate (eta) */
#define RND_SEED	1			/* random seed ( used srand ) */

#define IN_FILE "parity.in"		/* filename of input pattern */
#define TG_FILE "parity.tg"		/* filename of target */
#define WT_FILE "bparity.wt"		/* filename of weight data */
#define RMS_FILE "bparity.rms"	/* filename of error change */

typedef matrix<double> nn_weight_t;  /* type for a connection weight */
                                     /* also used to store a 'pattern' */
typedef vector<double> nn_uvalue_t;  /* type for net, output, etc. */


void randomize_matrix(nn_weight_t &mat, double iwr);
void read_file(nn_weight_t &pat, char* file);
double test_network(nn_weight_t &wih, nn_weight_t &who,
		nn_weight_t &in_p, nn_weight_t &tg_p);
void calc_layer(nn_weight_t &w,
		nn_uvalue_t &in_p, nn_uvalue_t &net, nn_uvalue_t &out);
void calc_output(nn_weight_t &wih, nn_weight_t &who,
		nn_uvalue_t &in_p, nn_uvalue_t &out_o);
void calc_output(nn_weight_t &wih, nn_weight_t &who,
		nn_uvalue_t &in_p,
		nn_uvalue_t &net_h, nn_uvalue_t &net_o,
		nn_uvalue_t &out_h, nn_uvalue_t &out_o);
double get_rms(nn_uvalue_t &x, nn_uvalue_t &y);
void prt_output(nn_weight_t &wih, nn_weight_t &who,
		nn_weight_t &in_p, nn_weight_t &out_p);
void train_network(nn_weight_t &wih, nn_weight_t &who,
		nn_weight_t &in_p, nn_weight_t &tg_p, double eta);
void calc_delta(nn_weight_t &wih, nn_weight_t &who,
		nn_uvalue_t &out_h, nn_uvalue_t &out_o,
		nn_uvalue_t &tg,
		nn_uvalue_t &del_h, nn_uvalue_t &del_o);
void adj_weight_layer(nn_weight_t &w, nn_uvalue_t out_pre, nn_uvalue_t del,
		double eta);
void adj_weight(nn_weight_t &wih, nn_weight_t &who,
		nn_uvalue_t &out_i, nn_uvalue_t &out_h,
		nn_uvalue_t &del_h, nn_uvalue_t &del_o,
		double eta);
void save_weight_layer(std::ostream &strm, nn_weight_t &w);
void save_weight(nn_weight_t &wih, nn_weight_t &who, char* file);

int
main(int argc, char* argv[])
{
	int  iter = 0;								/* iteration */
	nn_weight_t Wih(HID, INP + 1);			/* connection weight */
	nn_weight_t Who(OUT, HID + 1);
	nn_weight_t input_p(EX_NUM, INP);		/* input pattern */
	nn_weight_t target_p(EX_NUM, OUT);		/* target pattern */

	double ave_rms = MIN_ERROR + 1.0;			/* average of rms error */
	FILE *fp;
	double eta	= ETA;									/* set training rate */

	srand48(RND_SEED);
	randomize_matrix(Wih, IWR);
	randomize_matrix(Who, IWR);
//	std::cout << Wih << std::endl;
//	std::cout << Who << std::endl;

	read_file(input_p, IN_FILE);
	read_file(target_p, TG_FILE);
//	std::cout << input_p << std::endl;
//	std::cout << target_p << std::endl;

	std::ofstream rms_file(RMS_FILE);
	if (rms_file.fail()) {
		std::cerr << "Cannot open " << RMS_FILE << std::endl;
		exit(EXIT_FAILURE);
	}

	while ((ave_rms > MIN_ERROR) && (iter < MAX_ITER)) {
		ave_rms = test_network(Wih, Who, input_p, target_p);

		rms_file << iter << " " << ave_rms << std::endl;

		if (!(iter % 100)) {
			std::cout << "iteration: " << iter << std::endl;
			prt_output(Wih, Who, input_p, target_p);
			std::cout << Wih << std::endl;
			std::cout << Who << std::endl;
			save_weight(Wih, Who, WT_FILE);
		}

		train_network(Wih, Who, input_p, target_p, eta);

		iter ++;
	}
	save_weight(Wih, Who, WT_FILE);
	return EXIT_SUCCESS;
}

void
read_file(nn_weight_t &pat, char* fname)
{
	std::ifstream pstrm(fname);
	if (pstrm.fail()) {
		std::cerr << "Cannot open " << fname << std::endl;
		exit(EXIT_FAILURE);
	}
	for (int p = 0; p < pat.size1(); p ++) {
		for (int i = 0; i < pat.size2(); i ++) {
			double x;

			pstrm >> x;
			pat(p, i) = x;
//			std::cout << x << std::endl;
		}
	}
}

void
randomize_matrix(nn_weight_t &mat, double iwr)
{
	for (int row=0; row < mat.size1(); row ++) {
		for (int col=0; col < mat.size2(); col ++) {
			mat(row, col) = (drand48() * 2.0 - 1.0) * iwr;
		}
	}
}

void
calc_layer(nn_weight_t &w,
		nn_uvalue_t &in_p, nn_uvalue_t &net, nn_uvalue_t &out)
{
	in_p.resize(w.size2());
	in_p(in_p.size() - 1) = 1.0;  // bias
	net = prod(w, in_p);
	for (int h = 0; h < net.size(); h ++) {
		out(h) = tanh(net(h));
	}
}

void
calc_output(nn_weight_t &wih, nn_weight_t &who,
		nn_uvalue_t &in_p, nn_uvalue_t &out_o)
{
	nn_uvalue_t net_h;
	nn_uvalue_t net_o;
	nn_uvalue_t out_h(wih.size1());

	calc_output(wih, who, in_p, net_h, net_o, out_h, out_o);
}

void
calc_output(nn_weight_t &wih, nn_weight_t &who,
		nn_uvalue_t &in_p,
		nn_uvalue_t &net_h, nn_uvalue_t &net_o,
		nn_uvalue_t &out_h, nn_uvalue_t &out_o)
{
	calc_layer(wih, in_p, net_h, out_h);
	calc_layer(who, out_h, net_o, out_o);
}


double
test_network(nn_weight_t &wih, nn_weight_t &who,
		nn_weight_t &in_p, nn_weight_t &tg_p)
{
	nn_uvalue_t in;
	nn_uvalue_t tg;
	nn_uvalue_t out(who.size1());
	double sum_rms = 0.0;
	double ave_rms = 0.0;

	for (int p = 0; p < in_p.size1(); p ++) {
		double e_rms;
		in = matrix_row<nn_weight_t>(in_p, p);
		tg = matrix_row<nn_weight_t>(tg_p, p);
		calc_output(wih, who, in, out);
		e_rms = get_rms(out, tg);
		sum_rms += e_rms;
	}
	ave_rms = sum_rms / double(in_p.size1());
	return ave_rms;
}

double
get_rms(nn_uvalue_t &x, nn_uvalue_t &y)
{
	return norm_2(x - y) / sqrt(x.size());
}

void
prt_output(nn_weight_t &wih, nn_weight_t &who,
		nn_weight_t &in_p, nn_weight_t &tg_p)
{
	nn_uvalue_t in;
	nn_uvalue_t tg;
	nn_uvalue_t out(who.size1());
	double sum_rms = 0.0;

	for (int p = 0; p < in_p.size1(); p ++) {
		double e_rms;
		in = matrix_row<nn_weight_t>(in_p, p);
		tg = matrix_row<nn_weight_t>(tg_p, p);
		calc_output(wih, who, in, out);
		e_rms = get_rms(out, tg);
		sum_rms += e_rms;
		std::cout << p << ":";
		std::cout << in << " : " << tg << " : " << out << " : ";
		std::cout << e_rms << std::endl;
	}
	double ave_rms = sum_rms / double(in_p.size1());
	std::cout << "average of e_rms:" << ave_rms << std::endl;
}

void
train_network(nn_weight_t &wih, nn_weight_t &who,
		nn_weight_t &in_p, nn_weight_t &tg_p, double eta)
{
	nn_uvalue_t in;
	nn_uvalue_t tg;
	nn_uvalue_t net_h(wih.size1());
	nn_uvalue_t net_o(who.size1());
	nn_uvalue_t out_h(wih.size1());
	nn_uvalue_t out_o(who.size1());
	nn_uvalue_t del_h(wih.size1());
	nn_uvalue_t del_o(who.size1());

	for (int p = 0; p < in_p.size1(); p ++) {
		in = matrix_row<nn_weight_t>(in_p, p);
		tg = matrix_row<nn_weight_t>(tg_p, p);
		calc_output(wih, who, in, net_h, net_o, out_h, out_o);
		calc_delta(wih, who, out_h, out_o, tg, del_h, del_o);
		adj_weight(wih, who, in, out_h, del_h, del_o, eta);
	}
}

void
calc_delta(nn_weight_t &wih, nn_weight_t &who,
		nn_uvalue_t &out_h, nn_uvalue_t &out_o,
		nn_uvalue_t &tg,
		nn_uvalue_t &del_h, nn_uvalue_t &del_o)
{
	for (int o = 0; o < out_o.size(); o ++) {
		del_o(o) = (tg(o) - out_o(o)) * (1.0 - out_o(o) * out_o(o));
	}
	del_h = prod(trans(who), del_o);
	for (int h = 0; h < out_h.size(); h ++) {
		del_h(h) *= 1.0 - out_h(h) * out_h(h);
	}
}

void
adj_weight_layer(nn_weight_t &w, nn_uvalue_t out_pre, nn_uvalue_t del,
		double eta)
{
//	std::cout << "adj_weight_layer:begin" << std::endl;
//	std::cout << "w:" << w << std::endl;
//	std::cout << "out_pre:" << out_pre << std::endl;
//	std::cout << "del:" << del << std::endl;
	for (int nxt = 0; nxt < w.size1(); nxt ++) {
		for (int pre = 0; pre < w.size2(); pre ++) {
//			std::cout << "pre:" << pre << " nxt:" << nxt << std::endl;
			w(nxt, pre) += eta * del(nxt) * out_pre(pre);
		}
	}
}

void
adj_weight(nn_weight_t &wih, nn_weight_t &who,
		nn_uvalue_t &out_i, nn_uvalue_t &out_h,
		nn_uvalue_t &del_h, nn_uvalue_t &del_o,
		double eta)
{
	adj_weight_layer(wih, out_i, del_h, eta);
	adj_weight_layer(who, out_h, del_o, eta);
}

void
save_weight_layer(std::ostream &strm, nn_weight_t &w)
{
	for (int nxt = 0; nxt < w.size1(); nxt ++) {
		for (int pre = 0; pre < w.size2(); pre ++) {
			strm << w(nxt, pre) << std::endl;
		}
	}
	strm << std::endl;
}

void
save_weight(nn_weight_t &wih, nn_weight_t &who, char* file)
{
	std::ofstream wstr(file);
	if (wstr.fail()) {
		std::cerr << "Cannot open " << file << std::endl;
		exit(EXIT_FAILURE);
	}
	wstr << "#tutorial1" << std::endl;
	wstr << "3 "
		<< wih.size2() - 1 << " "
		<< who.size2() - 1 << " "
		<< who.size1() << std::endl;
	save_weight_layer(wstr, wih);
	save_weight_layer(wstr, who);
}

