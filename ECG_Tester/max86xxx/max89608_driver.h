
#ifndef _MAX86908_DRIVER_H_
#define _MAX86908_DRIVER_H_

#include "max86908_map.h"

void max86xx_init(void);
void max86xx_write_reg(uint8 cmd, uint8 data);
void max86xx_readfifodata(uint8 *buff, uint8 size);
uint8 max86xx_get_numsamples(void);
uint8 max86xx_read_reg(uint8 cmd);



#define I2C_STATUS_IDLE     0x00
#define I2C_STATUS_BUSY     0x01
#define I2C_STATUS_CMPLT    0x02


#endif
