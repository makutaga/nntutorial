/*
	3 Layer Neural Network Simulator ( Tutorial Version )
	Training Algorithm : Back Propagation ( Jump Every Time Version )

	This source is maintained in the 'svn (Subversion)' version control system.
	Based source in CVS repository is
	"Id: tutorial.c,v 1.9 2000/09/16 03:17:18 makutaga Exp" 

	$HeadURL$
*/
static char id[]="$Id$";


/*** Function List ***/
/* ---- level 2 NN Library ---- */
void   train_network( );   /* training NN (1 iter.)                        */
double test_network( );    /* test NN and calcurate rms error              */
void   prt_weight( );      /* print weight value                           */
void   prt_output( );      /* test NN and print NN's output                */
void   calc_output( );     /* calc. NN output vector for 1 pattern         */
/* ---- level 1 NN Library ---- */
void   forward_prop( );    /* forward propagation                          */
void   forward_tanh( );    /* forward propagation with tanh units         */
void   forward_linear( );  /* forward propagation with linear units (NOT YET)*/
void   calc_delta( );      /* calc. delta                                  */
void   adj_weight( );      /* alternate weights                            */
void   read_file( );       /* read a file into a vector                    */
void   save_weight( );     /* save weights                                 */
void   load_weight( );     /* load weights (DO IT YOURSELF)                */
/* ---- vector Library ---- */
void   randomize_vect( );  /* randomize vector                             */
double get_rms( );         /* calc. rms between 2 vectors                  */
void   prt_vector( );      /* fprintf a vector                             */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <errno.h>

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS	0
#endif
#ifndef EXIT_FAILURE
#define EXIT_FAILURE	1
#endif

/*
  If you don't need the bias units,
  comment out following line.
*/
#define BIAS

/* *** NOT SUPPORTED YET ***
  When you want to use linear output units,
  uncomment following line.
#define LINEAR_OUTPUT
*/

/*
	To get weight value from Weight Array ....
	    w[ from + to * (p_size + 1 )]
			from    : unit No. of previous layer
			to      : unit No. of next layer
			p_size  : No. of all units in previous layer ( except bias unit )
*/
#define WeightValue(w,p_size,n_size,from,to) (*((w)+(from)+(to)*(p_size+1)))

#define INP 3					/* number of input units  */
#define HID 2					/*           hidden units */
#define OUT 1					/*           output units */

#define EX_NUM 8           		/* number of input patterns */
#define MAX_ITER	10000		/* maximum of iteration */
#define MIN_ERROR	0.001		/* destination of error rms */

#define IWR			1.0			/* amplitude of initial weight */
#define ETA			0.1			/* training rate (eta) */
#define RND_SEED	1			/* random seed ( used srand48 ) */

#define EX_FILE "parity.in"		/* filename of input pattern */
#define TG_FILE "parity.tg"		/* filename of target */
#define WT_FILE "parity.wt"		/* filename of weight data */
#define RMS_FILE "parity.rms"	/* filename of error change */

/*========================================================================*/
int
main (int argc, char *argv[])
{    
	int  iter = 0;								/* iteration */
	double eta;									/* training rate */
	double Wih[(INP+1)*HID],  Who[(HID+1)*OUT]; /* weight */
	double input_p[EX_NUM*INP];					/* input data */
	double target_p[EX_NUM*OUT];				/* target data */
	double ave_rms = MIN_ERROR + 1.0;			/* average of rms error */
	FILE *fp;
	 
	eta	= ETA;									/* set training rate */
	srand48(RND_SEED);							/* set random seed */
    randomize_vect(Wih, (INP+1)*HID, IWR);	/* radomize weight value */
    randomize_vect(Who, (HID+1)*OUT, IWR);	/* radomize weight value */

    read_file(input_p, EX_NUM, INP, EX_FILE);	/* read input pattern data */
    read_file(target_p, EX_NUM, OUT, TG_FILE);/* read target data */

	if ((fp = fopen(RMS_FILE, "w")) == NULL) {
		perror(RMS_FILE);
		exit(EXIT_FAILURE);
	}

    while ((ave_rms > MIN_ERROR) && (iter < MAX_ITER)) {
		ave_rms = test_network(INP, HID, OUT, Wih, Who,
										input_p, target_p, EX_NUM);
		fprintf(fp, "%d %f\n", iter, ave_rms);
		fflush(fp);

		if (!(iter % 100))  {
			printf("iteration: %6d  Ave. of Error RMS: %f\n", iter, ave_rms);
			prt_output(INP, HID, OUT, Wih, Who, input_p, target_p, EX_NUM);
		}
		if (!(iter % 100)) {
			printf("iteration: %6d\n", iter);
			prt_weight(INP, HID, OUT, Wih, Who);
			save_weight(INP, HID, OUT, Wih, Who, WT_FILE);
		}

		train_network(INP, HID, OUT, Wih, Who,
							input_p, target_p, EX_NUM, eta);
		iter ++;							/* increment iteration counter */
	}
	save_weight(INP, HID, OUT, Wih, Who, WT_FILE);
	fclose(fp);

	return EXIT_SUCCESS;
}
/*---------------------------------------------------------------------*/
void
train_network(int inp, int hid, int out,
	double *wih, double *who, double *input_p, double *target_p,
	int p_num, double eta)
{
	int p;
	double  net_h[HID], net_o[OUT];
	double  out_h[HID], out_o[OUT];
	double  del_h[HID], del_o[OUT];

	for (p = 0 ; p < p_num ; p ++) {
		forward_prop(inp, hid, out, wih, who, &input_p[p*inp],
						net_h, net_o, out_h, out_o);
		calc_delta(inp, hid, out, wih, who,
						out_h, out_o, &target_p[p*out], del_h, del_o);
		adj_weight(inp, hid, out, wih, who,
						&input_p[p*inp], out_h, del_h, del_o, eta);
	}
}
/*----------------------------------------------------------------*/
double
test_network(int inp, int hid, int out,
	double *wih, double *who, double *t_input, double *t_target, int t_num)
{
	int t;
	double sum_rms = 0.0;
	double ave_rms;
	double output[OUT];

	for (t = 0 ; t < t_num ; t ++) {
		calc_output(inp, hid, out, wih, who, &t_input[t*inp], output);
		sum_rms	+= get_rms(output, &t_target[t*out], out);
	}
	ave_rms	= sum_rms / (double)t_num;

	return ave_rms;
}
/*---------------------------------------------------------------*/
void
prt_weight(int inp, int hid, int out, double *wih, double *who)
{
	int h, o;

	printf("Weight======\n");
	for (h = 0 ; h < hid ; h ++) {
		prt_vector(stdout, "%6.4f ", wih + h*(inp+1), inp+1);
		printf("\n");
	}
	for (o = 0 ; o < out ; o ++) {
		prt_vector(stdout, "%6.4f ", who + o*(hid+1), hid+1);
		printf("\n");
	}
}
/*---------------------------------------------------------------*/
void
prt_output(int inp, int hid, int out,
	double *wih, double *who, double *input_p, double *target_p,
	int p_num)
{
	int p;
	double output[OUT];
	double sum_rms=0.0, rms;

	for (p = 0 ; p < p_num ; p ++) {
		calc_output(inp, hid, out, wih, who, &input_p[p*inp], output);
		rms	= get_rms(output, &target_p[p*out], out);
		sum_rms	+= rms;
		printf("%3d:", p);
		prt_vector(stdout, " %6.3f ", input_p + p*inp, inp);
		printf(" : ");
		prt_vector(stdout, " %6.3f ", target_p + p*out, out);
		printf(" : ");
		prt_vector(stdout, " %6.3f ", output, out);
		printf(" : ");
		printf("%.4f\n", rms);
	}
}
/*----------------------------------------------------------------*/
void
calc_output(int inp,int hid, int out,
	double *wih, double *who, double *input_p, double *dest )
{
	double  net_h[HID], net_o[OUT];
	double  out_h[HID];

	forward_prop( inp, hid, out, wih, who, input_p, net_h, net_o, out_h, dest );
}
/*=================================================================*/
/*----------------------------------------------------------------
	Forward Propagation Function
	inp, hid, out : the number of units in input, hidden, output layer
	wih, who : array of weight value
	input_p : array of input pattern
	net_h, net_o : net value of hidden and output layer
    out_h, out_o : output of hidden units and output units
*/
void
forward_prop(int inp, int hid, int out,
	double *wih, double *who,
	double *input_p, double *net_h, double *net_o,
	double *out_h, double *out_o )
{
	forward_tanh(inp, hid, wih, input_p, net_h, out_h);
	forward_tanh(hid, out, who, out_h,   net_o, out_o);
}

void
forward_tanh(int prv, int nxt,
	double *w, double *prv_out, double *net, double *nxt_out)
{
	int p, n;
	double nt;

	for (n = 0 ; n < nxt ; n ++) {
		nt	= 0.0;
		for (p = 0 ; p < prv ; p ++) {
			nt	+= prv_out[p] * w[p + n*(prv + 1)];
		}
#ifdef BIAS
		nt	+= w[prv + n*(prv + 1)];
#endif
		net[n]		= nt;
		nxt_out[n]	= tanh(nt);
	}
}

/*----------------------------------------------------------------*/
void
calc_delta(int inp, int hid, int out,
	double *wih, double *who, double *out_h, double *out_o,
	double *tg, double *del_h, double *del_o )
{
	int h, o;
	double d;

	for (o = 0 ; o < out ; o ++) {
		del_o[o] = (tg[o] - out_o[o]) * (1.0 - out_o[o] * out_o[o]);
	}
	for (h = 0 ; h < hid ; h ++) {
		d	= 0.0;
		for (o = 0 ; o < out ; o ++) {
			d += del_o[o] * who[h + o*(hid + 1)];
		}
		del_h[h] = d * (1.0 - out_h[h] * out_h[h]);
	}
}
/*----------------------------------------------------------------*/
void
adj_weight(int inp, int hid, int out,
	double *wih, double *who,
	double *out_i, double *out_h, double *del_h, double *del_o, double eta)
{
	int i, h, o;

	for (o = 0 ; o < out ; o ++) {
		for (h = 0 ; h < hid ; h ++) {
			who[h + o * (hid + 1)] += eta * del_o[o] * out_h[h];
		}
#ifdef BIAS
		who[hid + o * (hid + 1)] += eta * del_o[o];
#endif 
	}

	for (h = 0 ; h < hid ; h ++) {
		for (i = 0 ; i < inp ; i ++) {
			wih[i + h * (inp + 1)] += eta * del_h[h] * out_i[i];
		}
#ifdef BIAS
		wih[inp + h * (inp + 1)] += eta * del_h[h];
#endif
	}
}
/*------------------------------------------------------------
  Read file into array
*/
void
read_file(double *dat, int p_num, int u_num, char *file)
{ 
	int p, u;
	FILE *fp;

	if ((fp = fopen(file, "r")) == NULL) {
		perror(file);
		exit(EXIT_FAILURE);
	}

	for (p = 0 ; p < p_num ; p ++) {
		for (u = 0 ; u < u_num ; u ++) {
			fscanf(fp, "%lf", dat++);
		}
	}

	fclose(fp);
}

/*--------------------------------------------------------------------
  Save Weight Value
*/
void
save_weight(int inp, int hid, int out,
	double *wih, double *who, char *file )
{
	int i, h, o;
	FILE *fp;

	if ((fp = fopen(file, "w")) == NULL) {
		perror(file);
		exit(EXIT_FAILURE);
	}

	fprintf(fp, "#tutorial1\n");					/* file ID */
	fprintf(fp, "3 %d %d %d\n", inp, hid, out);		/* NN configuration */
	for (h = 0 ; h < hid ; h ++) {
		for (i = 0 ; i <= inp ; i ++) {
			fprintf(fp, "%e\n", wih[i + h * (inp + 1)]);
		}
	}
	fprintf(fp, "\n");
	for (o = 0 ; o < out ; o ++) {
		for (h = 0 ; h <= hid ; h ++) {
			fprintf(fp, "%e\n", who[h + o * (hid + 1)]);
		}
	}
	fclose(fp);
}
/*-----------------------------------------------------------------------*/
void
load_weight(int inp, int hid, int out,
	double *wih, double *who, char *file)
{
		/* write this function yourself !! */
}

/*=============== several convenient vector function ================*/
/*----------------------------------------------------------------
  Randomize n-dimension vector
*/
void
randomize_vect(double *w, int n, double iwr )
{   
	int i;
	double drand48();

	for (i = 0 ; i < n ; i ++) {  
		*(w ++)  = (drand48( ) * 2.0 - 1.0) * iwr;
	}
}
/*----------------------------------------------------------------
  Calcurate RMS between two vector
*/
double
get_rms(double *x, double *y, int n)
{

	int i;
	double e;
	double sum = 0.0;

	for (i = 0 ; i < n ; i ++) {
		e	= *(x ++) - *(y ++);
		sum	+= e * e;
	}
	return (sqrt( sum / (double)n ));
}
/*--------------------------------------------------------------*/
void
prt_vector(FILE *fp, char *fmt, double *dat, int n)
{
	for ( ; n > 0 ; n -- ) {
		fprintf(fp, fmt, *(dat ++));
	}
}

