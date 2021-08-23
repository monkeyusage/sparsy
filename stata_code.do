
cd "e:\RA_matrix"		/* define the file directory */

* this loop is to subset the data
* matrix raw data.dta: main input data file has 4 variables: division, year, nclass, firm
forvalues z = 1975(1)1980 {
use "matrix raw data.dta",clear  
keep if year == `z' |  year == `z'-1 |  year == `z'-2	/* Note: "|" is "or" operator	*/
save "data1_`z'.dta",replace 
}

clear all
set more off		/* STATA specific command */
set maxvar 32767	/* STATA specific command to set max # of variables */
set matsize 11000	/* STATA specific command to set max matrix size */

forvalues z = 1975(1)1980  {
use "data1_`z'.dta",clear 
sort nclass
/* create sequential number of nclass */
gen tclass=1	/* tclass is the labelng of nclass in sequence; Note: gen and egen are STATA command to generate new variables */
replace tclass=tclass[_n-1]+1*(nclass~=nclass[_n-1]) if nclass[_n-1]~=. /* Note: [_n-1] is lag 1; ~= is not equal; "." is missing */
egen TECH=max(tclass)  	/* create TECH to be the max of tclass */ 
global define tech=TECH  	/* global macro defining tech=TECH ; ; to be called up later as $tech */
egen divByFirm=count(division),by(firm) 	/* get total # of divisions by firm */
egen divByFirmTclass=count(division),by(firm tclass) 	/* get total # of divisions by firm & tclass */
fillin firm tclass  	/* rectangularize firm & tclass by filling in missing observations */
drop _f  	/* drop temporary variable _f created by fillin in line 26 */
sort firm tclass
drop if firm==firm[_n-1]&tclass==tclass[_n-1]  /* drop duplicate observations in terms of firm and tclass */
gen double subsh=100*(divByFirmTclass/divByFirm)  /* generate variable subsh in double format */
keep firm tclass subsh
qui summarize tclass  /* summarize tclass for line 33 */
global tech=r(max)  /* define global macro tech to be max of tclass generated from line 32 */
replace subsh=0 if subsh==.
reshape wide subsh, i(firm) j(tclass) /* reshaping the data from long to wide, j is existing variable   */
compress
save "data2_`z'.dta",replace
clear all
set more off		/* STATA specific command */
set maxvar 32767	/* STATA specific command to set max # of variables */
set matsize 11000	/* STATA specific command to set max matrix size */
use "data2_`z'.dta",clear

capture drop nclass subcat tclass TECH total
sort firm
gen num=1
replace num=num[_n-1]+1*(firm~=firm[_n-1]) if firm[_n-1]~= ""  /* generate distinct num for each firm  */
sort num
preserve
egen tag=tag(num)   /* It tags just one observation in each distinct group defined by varlist, num.   */
keep num firm
save "num_gvkey_`z'", replace
rename num num_
rename firm firm_
save "num_gvkey__`z'", replace
restore
egen NUM=max(num) /* create NUM to be the max # of num */ 
qui summarize num
global num=r(max) /* define global macro num to be max # of num from line 57; to be called up later as $num */

*Generates a matrix of all the shares in dimensions (firm, class) 
mkmat subsh*,mat(subsh)    /* create matrix named "subsh" from variable subsh created above */
matrix normsubsh=subsh

*Var is a (class,class) matrix of the correlations between classes. Used for Mahalanobis distance measures
matrix var=subsh'*subsh
matrix basevar=var
forv i=1(1)$tech {
forv j=1(1)$tech {
matrix var[`i',`j']=var[`i',`j']/(basevar[`i',`i']^(1/2)*basevar[`j',`j']^(1/2))
}
}

*Standard is a (num,num) matrix of the correlations between firms over tech classes
matrix basestandard=subsh*subsh'
forv j=1(1)$tech {
forv i=1(1)$num {
matrix normsubsh[`i',`j']=subsh[`i',`j']/(basestandard[`i',`i']^(1/2))
}
}
matrix standard=normsubsh*normsubsh'
matrix covstandard=subsh*subsh'
save "temp_`z'",replace



global X=ceil($num/500)*500 - 500

* BL: TOO LONG, NEED TO SPLIT UP MATRICES




forv mal=0(1)1{
u "temp_`z'",clear

*Generate the Malhabois measure
if `mal'==1 {
matrix mal_corr=normsubsh*var*normsubsh'
matrix standard=mal_corr
matrix covmal_corr=subsh*var*subsh'
matrix covstandard=covmal_corr
}
*Convert back into scalar data
keep firm
sort firm
local J=$X+1
forv j=1(500)`J' {
preserve
local j2=`j'+499
if `j'==`J' {
	local j2 .y6
}
matrix covstandardj`j'=covstandard[1...,`j'..`j2']
matrix standardj`j'=standard[1...,`j'..`j2']
svmat standardj`j',n(standard)   /* save matrix column "standard" as variable standardj`j' */
svmat covstandardj`j',n(covstandard)   /* save matrix column "covstandard" as variable covstandardj`j' */
compress
reshape long standard covstandard,i(firm) j(num_)   /* reshaping from wide to long, j () is  new variable */
capture drop subsh*
rename *standard *tec	/* renaming variables ending with "standard" to end with "tec"	*/ 
replace num_ = `j'+num_-1
sort firm num_
*convert to integers to reduce memory size - renormalize later
foreach var in tec covtec {
capture replace `var'=100*round(`var',0.01)
}
compress
if `mal'==1 {
rename *tec mal*tec		/* renaming variables ending with "tec" to start with "mal"	*/ 
save "output_short70_mal_newj`j'_`z'",replace
}
else {
save "output_short70_newj`j'_`z'",replace
}
restore
}
}
foreach f in output_short70_new output_short70_mal_new { /*there are two types of outputs, output_short70_new based on correlation measure, output_short70_mal_new  baseed on Mahalanobis measure*/
clear
forv j=1(500)`J' {
	append using "`f'j`j'_`z'"
}
sort firm num_
merge m:1 num_ using "num_gvkey__`z'"
assert _m==3
drop _m
save "`f'_`z'", replace
* clean up
forv j=1(500)`J' {
	erase "`f'j`j'_`z'.dta"  /*delete redundant data files */
}
}
}

clear all

forvalues z = 1975(1)1980 {
macro define metrics "tec covtec maltec malcovtec" /* define four spillover measures to be called as $metrics below */

*THIS SECTION MERGES THE CORRELATION OUTPUT FILES TOGETHER
clear
use output_short70_mal_new_`z' ,clear
sort firm firm_
save output_short70_mal_new_`z' ,replace 

use output_short70_new_`z'
sort firm firm_
merge firm firm_ using output_short70_mal_new_`z' 
keep if _==3
drop _merge
sort firm firm_

sort firm
order $metrics firm firm_
keep $metrics firm firm_

cap lab var tec "JAFFE Closeness in Technology Space (TECH)"

cap lab var covtec "Covariance Closeness in Technology Space (TECH)"

cap lab var maltec "Mahalanobis Closeness in Technology Space (TECH)"

cap lab var malcovtec "Covariance Mahalonobis Closeness in Technology Space (TECH)"


*drop if firm == firm_
drop if firm==""
drop if firm_==""
sort firm_
*convert to integers to reduce memory size - renormalize later
foreach var in $metrics {
replace `var'=100*round(`var',0.01)
}
compress



foreach var in $metrics {
egen spill`var'=sum(`var'*(1/100)*(firm~=firm_)), by(firm) /*generate spillover measures for each (firm firm_) pair, and aggregate at firm level*/
drop `var'
}
keep if firm == firm_
drop firm_
sort firm 
gen year = `z'
foreach var in $varlist {
gen l`var'=log(spill`var')
}
compress
save spill_output_`z',replace
clear all
}





