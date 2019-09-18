/* Copyright 2019 Sree Harsha Angara. All Rights Reserved.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mbed.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "constants.h"
#include "output_handler.h"
#include "model_data.h"
#include "max86908_map.h"
#include "max89608_driver.h"

#define NUM_SAMPLES_NN_INPUT    320

typedef struct __attribute__((packed)) {
    uint8_t data_rdy;
    uint8_t *read_buff;
    uint8_t *read_buff_shadow;
    int32_t *corrected_input;
} Thread_struct_t;

Thread_struct_t thread_struct;

InterruptIn edge_det(P9_5);
DigitalOut dbg_1(P0_2);
DigitalOut dbg_2(P5_6);

Thread thread;
RawSerial kp_uart(UART_TX, UART_RX);


uint8_t *wr_ptr;
volatile uint8_t int_det_flag = 0;

uint8_t read_buf[NUM_SAMPLES_NN_INPUT*3], read_buf_shadow[NUM_SAMPLES_NN_INPUT*3];
int32_t corrected_input[NUM_SAMPLES_NN_INPUT];

// float test_input_array[320] = {0.2135593,0.2203390,0.2237288,0.1966102,0.1694915,0.1694915,0.1593220,0.1525424,0.1389831,0.1389831,0.1423729,0.1491525,0.1457627,0.1491525,0.1423729,0.1322034,0.1288136,0.1322034,0.1322034,0.1254237,0.1288136,0.1220339,0.1389831,0.1355932,0.1389831,0.1457627,0.1288136,0.1220339,0.1322034,0.1389831,0.1423729,0.1389831,0.1254237,0.1152542,0.1186441,0.1050847,0.0745763,0.0610170,0.0406780,0.0169492,0.0000000,0.0203390,0.0813559,0.1593220,0.2542373,0.3525424,0.5220339,0.6949152,0.8508475,0.9559322,1.0000000,0.9762712,0.8745763,0.6949152,0.4745763,0.2881356,0.1796610,0.0983051,0.0576271,0.0406780,0.0542373,0.0745763,0.0983051,0.0983051,0.1016949,0.0983051,0.1050847,0.1050847,0.1050847,0.0949153,0.1016949,0.1016949,0.1016949,0.1084746,0.0949153,0.0847458,0.0847458,0.0915254,0.0949153,0.0983051,0.0881356,0.0949153,0.1016949,0.1084746,0.0983051,0.0983051,0.0881356,0.0847458,0.0847458,0.0983051,0.0983051,0.0949153,0.0881356,0.0813559,0.0915254,0.0847458,0.0881356,0.0847458,0.0813559,0.0847458,0.0915254,0.0915254,0.0949153,0.0983051,0.0881356,0.0847458,0.0983051,0.0983051,0.1016949,0.1016949,0.0847458,0.0881356,0.0949153,0.0983051,0.1084746,0.0983051,0.0949153,0.0847458,0.0915254,0.0983051,0.0983051,0.1016949,0.0881356,0.0813559,0.0813559,0.0915254,0.0949153,0.0881356,0.0813559,0.0779661,0.0779661,0.0847458,0.0813559,0.0745763,0.0779661,0.0677966,0.0711864,0.0711864,0.0779661,0.0711864,0.0576271,0.0542373,0.0576271,0.0677966,0.0610170,0.0576271,0.0542373,0.0440678,0.0508475,0.0542373,0.0576271,0.0542373,0.0440678,0.0406780,0.0508475,0.0576271,0.0610170,0.0644068,0.0677966,0.0644068,0.0847458,0.0949153,0.1118644,0.1152542,0.1186441,0.1152542,0.1322034,0.1423729,0.1491525,0.1457627,0.1389831,0.1389831,0.1457627,0.1525424,0.1593220,0.1593220,0.1457627,0.1457627,0.1491525,0.1559322,0.1559322,0.1627119,0.1457627,0.1389831,0.1559322,0.1525424,0.1593220,0.1593220,0.1457627,0.1423729,0.1525424,0.1559322,0.1491525,0.1559322,0.1491525,0.1491525,0.1525424,0.1627119,0.1661017,0.1559322,0.1525424,0.1457627,0.1491525,0.1593220,0.1491525,0.1491525,0.1491525,0.1389831,0.1491525,0.1457627,0.1491525,0.1423729,0.1355932,0.1322034,0.1355932,0.1355932,0.1389831,0.1423729,0.1288136,0.1254237,0.1288136,0.1355932,0.1389831,0.1389831,0.1288136,0.1288136,0.1288136,0.1457627,0.1389831,0.1355932,0.1322034,0.1220339,0.1355932,0.1389831,0.1389831,0.1355932,0.1220339,0.1186441,0.1220339,0.1254237,0.1322034,0.1288136,0.1288136,0.1288136,0.1288136,0.1389831,0.1389831,0.1355932,0.1288136,0.1254237,0.1254237,0.1322034,0.1389831,0.1288136,0.1288136,0.1220339,0.1254237,0.1322034,0.1389831,0.1322034,0.1322034,0.1322034,0.1389831,0.1355932,0.1389831,0.1355932,0.1288136,0.1288136,0.1322034,0.1389831,0.1491525,0.1627119,0.1593220,0.1627119,0.1694915,0.1762712,0.1830508,0.1830508,0.1830508,0.1898305,0.1932203,0.2000000,0.2101695,0.2169491,0.2000000,0.1966102,0.2000000,0.1966102,0.2033898,0.1966102,0.1932203,0.1762712,0.1762712,0.1830508,0.1932203,0.2101695,0.2135593,0.2067797,0.1830508,0.1796610,0.1728813,0.1694915,0.1559322,0.1457627,0.1322034,0.1389831,0.1355932,0.1355932,0.1220339,0.1118644,0.1186441,0.1288136,0.1254237,0.1220339,0.1152542,0.1084746,0.1050847,0.1118644,0.1152542,0.1152542
// };



int32_t process_filter(int32_t input)
{
     /* 60Hz band stop, 200Hz sample rate coeff */
    // float coefficients_xn[3] = {0.662460565567017,0.102763175964355,0.662460565567017};
    // float coefficients_yn[2] = {0.102763175964355, 0.324919700622559};
    /* 10Hz low pass, 200Hz sample rate coeff */
    float coefficients_xn[3] = {0.0200834274291992, 0.0401668548583984, 0.0200834274291992};
    float coefficients_yn[2] = {-1.56101822853088, 0.641351699829102};

    float coefficients_xn_1[3] = {0.107686042785645, 0.21537184715271, 0.107686042785645};
    float coefficients_yn_1[2] = {-1.47967433929443, 0.555821657180786};

    static int32_t xn_1 = 0, xn_2 = 0, yn_1 = 0, yn_2 = 0;
    static int32_t xn_1_1 = 0, xn_1_2 = 0, yn_1_1 = 0, yn_1_2 = 0;

    int32_t output_1, output_2;

    /* Biquad -1 */
    output_1 = input*coefficients_xn[0] + xn_1 * coefficients_xn[1]  + xn_2 * coefficients_xn[2] - yn_1 * coefficients_yn[0] - yn_2 *coefficients_yn[1];
    xn_2 = xn_1;
    xn_1 = input;
    
    yn_2 = yn_1;
    yn_1 = output_1;

    input = output_1;
    
    /* Biquad -2 */
    output_2 = input*coefficients_xn_1[0] + xn_1_1 * coefficients_xn_1[1]  + xn_1_2 * coefficients_xn_1[2] - yn_1_1 * coefficients_yn_1[0] - yn_1_2 *coefficients_yn_1[1];
    xn_1_2 = xn_1_1;
    xn_1_1 = input;
    
    yn_1_2 = yn_1_1;
    yn_1_1 = output_2;
    

    return(output_2);

}

void TF_thread(Thread_struct_t *thread_struct) 
{
    uint8_t *read_ptr;
    uint32_t loop;
    uint8_t header[2] = {0x17, 0xAB};
    float testout[2] = {0.1, 0.9};
    float normalized_input[320];

    // Set up tflite logging
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    const tflite::Model* model = ::tflite::GetModel(converted_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      error_reporter->Report(
          "Model provided is schema version %d not equal "
          "to supported version %d.\n",
          model->version(), TFLITE_SCHEMA_VERSION);
      while(1);
    }

    // This pulls in all the operation implementations we need
    tflite::ops::micro::AllOpsResolver resolver;

    // Create an area of memory to use for input, output, and intermediate arrays.
    // Finding the minimum value for your model may require some trial and error.
    // const int tensor_arena_size = 2 * 1024;
    uint8_t tensor_arena[6*1024];

    // Build an interpreter to run the model with
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                        sizeof(tensor_arena), error_reporter);

    // Allocate memory from the tensor_arena for the model's tensors
    interpreter.AllocateTensors();

    // Obtain pointers to the model's input and output tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    while(1)
    {
        
        if(thread_struct->data_rdy)
        {
            int32_t max=0, min=0;

            if(wr_ptr == read_buf_shadow)
            {
               read_ptr = read_buf; 
            }
            else
            {
                read_ptr = read_buf_shadow;
            }
            

            for(loop = 0; loop < NUM_SAMPLES_NN_INPUT ; loop++)
            {   
                uint8_t curr_samp_hi =  *(read_ptr + loop*3);
                uint8_t curr_samp_mid = *(read_ptr + loop*3 + 1);
                uint8_t curr_samp_low = *(read_ptr + loop*3 + 2);

                
                /* Clear any old data */
                corrected_input[loop] = 0;
                
                /* Check for 18th bit high */
                if(curr_samp_hi & (0x02))
                {   
                    /* Extend sign to full 32-bits */
                    corrected_input[loop] |= 0xFFFE0000; 
                }

                corrected_input[loop] |= (uint32_t) (curr_samp_hi << 16);
                corrected_input[loop] |= (uint32_t) (curr_samp_mid << 8);
                corrected_input[loop] |= (uint32_t) (curr_samp_low);
            
                corrected_input[loop] = process_filter(corrected_input[loop]);

                if(corrected_input[loop] > max)
                {
                    max = corrected_input[loop];
                }

                if(corrected_input[loop] < min)
                {
                    min = corrected_input[loop];
                }
            }

            /* Normalize input */
            for(loop = 0; loop < NUM_SAMPLES_NN_INPUT ; loop++)
            {
                  normalized_input[loop] = ((float) (corrected_input[loop] - min))/(max-min);
            }

            /* Load data into test array */
            input ->data.f = normalized_input;

            dbg_2.write(1); 

            // Run inference, and report any error
            TfLiteStatus invoke_status = interpreter.Invoke();
            if (invoke_status != kTfLiteOk) {
              error_reporter->Report("Invoke failed on input");
              continue;
            }
            dbg_2.write(0);


            testout[0] = output->data.f[0];
            testout[1] = output->data.f[1];
            
            /* Send output */
            kp_uart.putc(header[0]);
            kp_uart.putc(header[1]);

            for(loop = 0; loop < sizeof(testout) ; loop++)
            {
                
                kp_uart.putc((uint8_t)*((uint8_t *)&testout + loop));
            }

            /* Send input */
            for(loop = 0; loop < sizeof(corrected_input) ; loop++)
            {
                
                kp_uart.putc((uint8_t)*((uint8_t *)&corrected_input + loop));
            }

            thread_struct->data_rdy = 0;

        }

        ThisThread::yield();
    }
}

void int_det(void)
{
    int_det_flag = 0xFF;
}


int main() {

    thread_struct.data_rdy = 0;
    uint32_t wr_offset = 0;
    wr_ptr = read_buf;

    int_det_flag = 0;
    /* Set speed to 115200 */
    kp_uart.baud(115200);

    /* Setup edge detect */
    edge_det.mode(PullUp);
    edge_det.fall(&int_det);

    thread.start(callback(TF_thread,&thread_struct));
    
    max86xx_init();

    while(1)
    {
        
        if(int_det_flag)
        {
            
            uint8_t num_samples = max86xx_get_numsamples();

            /* Check if we aren't overflowing the available buffer */
            if((wr_offset + num_samples*3) > (sizeof(read_buf)))
            {
                /* Read the exact number of samples to fill buffer, get rest into new buffer next time */
                num_samples = (sizeof(read_buf) - wr_offset)/3;
            }

            max86xx_readfifodata(wr_ptr + wr_offset, num_samples*3);
            wr_offset += num_samples*3;
            
            int_det_flag = 0;
            if(wr_offset == (sizeof(read_buf)))
            {
                wr_offset = 0;
                if(wr_ptr == read_buf_shadow)
                {               
                    dbg_1.write(1);
                    wr_ptr = read_buf;
                }
                else 
                {
                    dbg_1.write(0);

                    wr_ptr = read_buf_shadow;
                }

                thread_struct.data_rdy = 1;

            }
            
        }
    
    }


}
