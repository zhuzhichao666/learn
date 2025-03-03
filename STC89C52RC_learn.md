**流水灯**

`#include <reg52.h>`
`unsigned char LED[8] = {0xFE, 0xFD, 0xFB, 0xF7, 0xEF, 0xDF, 0xBF, 0x7F};`
`void delay(unsigned int t)`
`{`
		`t *= 30000;`
		`while(t--);`
`}`
`void main()`
`{`
	`unsigned char i;`
	`while(1){`
	`for(i = 0;i < 8;i++){`
		`P0 = LED[i];`
		`delay(1);`
	`}` 
	`}`
`}`

