#include "mbed.h"
#include "max89608_driver.h"

/* Quick and dirty driver */

I2C i2c(P9_1 , P9_0 ); 

#define PPG_TEST

void max86xx_init(void)
{

    i2c.frequency(400000);
    /* Reset Chip */
    max86xx_write_reg(MAX86XXX_REG_SYSTEM_CTRL,0x01);
    wait_ms(2);

    /* Set to 400sps */
    max86xx_write_reg(MAX86XXX_REG_ECG_ETI_CFG1, 0x02);

    // max86xx_write_reg(MAX86XXX_REG_ECG_ETI_CFG_2,0x11); //hidden register at ECG config 2, per JY's settings
    // max86xx_write_reg(MAX86XXX_REG_ECG_ETI_CFG_4,0x02); //ECG config 4 per JY's settings

    /* Set gains */
    max86xx_write_reg(MAX86908_REG_ECG_ETI_CFG_3, 0x0A);


    /* Set Fifo Full to 12 -> 20 samples */
    max86xx_write_reg(MAX86908_REG_FIFO_CFG, 0x05);


    /* Enable FD1  as ECG */
    max86xx_write_reg(MAX86XXX_REG_FIFO_DATA_CTRL_1, 0x09);

    /* Enable Interrupt */
    max86xx_write_reg(MAX86908_REG_INT_ENABLE_1, 0x80);

    /* Enable FIFO */
    max86xx_write_reg(MAX86XXX_REG_SYSTEM_CTRL, MAX86XXX_MASK_SYSTEM_FIFO_EN);


    // max86xx_write_reg(MAX86908_REG_FIFO_WR_PTR, 0);
    // max86xx_write_reg(MAX86908_REG_OVF_CNT, 0);
    // max86xx_write_reg(MAX86908_FIFO_RD_PTR, 0);
}


void max86xx_write_reg(uint8 cmd, uint8 data)
{
    uint8 buff[2];
    buff[0] = cmd;
    buff[1] = data;
    i2c.write( MAX86XXX_SLAVE_ADDR, (const char *) buff, 2);

}

static inline struct i2c_s *cy_get_i2c_new(i2c_t *obj)
{
#if DEVICE_I2C_ASYNCH
    return &(obj->i2c);
#else
    return obj;
#endif
}

uint8 max86xx_read_reg(uint8 cmd)
{
    uint8 rd_cmd[1]; 
    uint8 buff[1];

    rd_cmd[0] = cmd;

    i2c.write( MAX86XXX_SLAVE_ADDR, (const char *)rd_cmd, 1, 1);
    i2c.read(MAX86XXX_SLAVE_ADDR, (char *) buff, 1);
    return(buff[0]);
}


uint8 max86xx_get_numsamples(void)
{
    uint8_t write_ptr, read_ptr;
    int16_t temp;

    write_ptr = max86xx_read_reg(MAX86XXX_REG_FIFO_WRITE_PTR);
    read_ptr = max86xx_read_reg(MAX86908_FIFO_RD_PTR);
    temp = write_ptr - read_ptr;
    
    if(temp < 0)
    {
        return(write_ptr - read_ptr + 32);
    }
    else
    {
        return(write_ptr - read_ptr);
    }
    
}


void max86xx_readfifodata(uint8 *buff, uint8 size)
{
    uint8 cmd[1] = {MAX86XXX_REG_FIFO_DATA};
    i2c.write( MAX86XXX_SLAVE_ADDR, (const char *)cmd, 1, true);
    i2c.read(MAX86XXX_SLAVE_ADDR, (char *) buff, size);
    /* Clear status */
    max86xx_read_reg(MAX86908_REG_INT_STATUS_1);
    

}


