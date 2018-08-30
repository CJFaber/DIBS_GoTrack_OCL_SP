// Implements the GoTrack -> csv transform using three different Kernels
// 		and pipes to create a streaming transform. 
//
// Author: Clayton Faber

#define MAX_LINE_SIZE 128
#define DELIMITER ','
#define NEWLINE '\n'
#define DELIM_COUNT 4
#define CHAN_DEPTH 20

//Structure to pass in the pipes
struct CSVline {
    char line[MAX_LINE_SIZE];
    short endIndex;
    short delimIndex[DELIM_COUNT];
} typedef CSVline;

//Channel for getting to the end of the doc will stop the delimiters loop
channel bool end_of_doc ;
//Channel for finishing find delimiters notifies the transform
channel bool end_of_delim ;
//Channel for holding on find delimiters set by FindNewLines
channel bool start_find_delim ;
//Channel for holding on transform set by FindDelimiters
channel bool start_transform ;
//Channel for FindNewLines -> FindDelimiters
channel CSVline NL_to_DL __attribute__((depth(CHAN_DEPTH))) ; 
//Channel for FindDelimiters -> Transform
channel CSVline DL_to_Trans __attribute__((depth(CHAN_DEPTH))) ; 

__kernel void FindNewLines(
    __global const char* restrict inputBuffer,
	unsigned long BufLen)
{
	unsigned long i;
	unsigned long start = 0;
	bool done = 1;
	CSVline curLine;
	//#pragma unroll MAX_LINE_SIZE					   //Don't think I can unroll with a write			
	write_channel_altera(start_find_delim, done);
	for(i = 0; i < BufLen; ++i){
		curLine.line[(i-start)] = inputBuffer[i];      //add endline in csvline
		if(inputBuffer[i] == NEWLINE){
			curLine.endIndex = (i-start);
			write_channel_altera(NL_to_DL, curLine);
			start = i;
		}
	}
	write_channel_altera(end_of_doc, done);			
}


__kernel void FindDelimiters(){
	bool sig_valid = 0; //end of doc valid flag
	bool sig_start = 0;
	bool data_valid = 0; //valid data flag
	bool processing_finish = 0;
	bool done = 1; 
	unsigned short commaIter;
	unsigned j = 0;
	CSVline curLine;
	sig_start = read_channel_altera(start_find_delim);
	write_channel_altera(start_transform, done);
	while(!sig_valid){ 
            curLine = read_channel_nb_altera(NL_to_DL, &data_valid);	//Non-blocking call
			commaIter = 0;
			//Find delimiters
			j = 0;
			if(data_valid){
				#pragma unroll 16
	        	while((j <= curLine.endIndex) && (commaIter < DELIM_COUNT)){
    	        	if(curLine.line[j] == DELIMITER){
        	        	curLine.delimIndex[commaIter] = j;
            	    	++commaIter;
            		}
					++j;
        		}
			write_channel_altera(DL_to_Trans, curLine);	
        	}
		//Check and see if we got the stop signal
		if(!data_valid){
			processing_finish = read_channel_nb_altera(end_of_doc, &sig_valid);
		}
	}
	write_channel_altera(end_of_delim, done);
}

__kernel void Transform(
	__global char* restrict outputBuffer,
	unsigned long BufLen)
{
	bool sig_valid = 0; 		//end of delim valid flag
	bool sig_start = 0;			
	bool data_valid = 0; 		//valid data flag
	bool processing_finish = 0; //processing finish flag
	unsigned long j = 0;
	CSVline curLine;
	sig_start = read_channel_altera(start_transform);
	while(!sig_valid && (j < BufLen)){
		curLine = read_channel_nb_altera(DL_to_Trans, &data_valid);
		if(data_valid){
			#pragma unroll 16
			for(unsigned i = 0; i < curLine.delimIndex[2]; ++i){
				outputBuffer[j] = curLine.line[i];
				++j;
			}
			outputBuffer[j] = NEWLINE;
			++j;
		}
		if(!data_valid){
			processing_finish = read_channel_nb_altera(end_of_delim, &sig_valid);
		}
	}
	outputBuffer[j] = 0x1a; 	//EOF flag
}

