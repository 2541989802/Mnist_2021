import {b1,b2,b3,b4,b5,w1,w2,w3,w4,w5} from "./variables.js";

/*console.log(b1.length,b1[0].length,b1[0][0].length)
console.log(b2.length,b2[0].length,b2[0][0].length)
console.log(b3.length,b3[0].length)
console.log(b4.length,b4[0].length)
console.log(b5.length,b5[0].length)
console.log(w1.length,w1[0].length,w1[0][0].length,w1[0][0][0].length)
console.log(w2.length,w2[0].length,w2[0][0].length,w2[0][0][0].length)
console.log(w3.length,w3[0].length)
console.log(w4.length,w4[0].length)
console.log(w5.length,w5[0].length)*/

export function forward(In){
    In =In.map((value)=>{
        return value.map((value)=>{
            return value.map((value)=>{
                return value/255.0
            })
        })
    })
    let x1 = convolute(In,w1)
    x1 = x1.map((value,index)=>{
        return value.map((value)=>{
            return value.map((value)=>{
                return relu(value+b1[index][0][0])
            })
        })
    })
    x1 = pool(x1)

    let x2 = convolute(x1,w2)
    x2 = x2.map((value,index)=>{
        return value.map((value)=>{
            return value.map((value)=>{
                return relu(value+b2[index][0][0])
            })
        })
    })
    x2 = pool(x2)
    x2 = reshape(x2)
    
    let x3 = matmul2D(x2,w3)
    x3 = x3.map((value)=>{
        return value.map((value,index)=>{
            return relu(value+b3[0][index])
        })
    })
    let x4 = matmul2D(x3,w4)
    x4 = x4.map((value)=>{
        return value.map((value,index)=>{
            return relu(value+b4[0][index])
        })
    })
    let x5 = matmul2D(x4,w5)
    x5 = x5.map((value)=>{
        return value.map((value,index)=>{
            return relu(value+b5[0][index])/100
        })
    })
    x5 = softmax(x5)
    return select(x5)
}
function matmul2D(a,b){
    let I = a.length
    let J = a[0].length
    let K = b[0].length
    if(J!=b.length){
        console.log("a and b are not match in dimension")
        return
    }
    let matmul_result = new Array()
    for(let i=0;i<I;i++){
        matmul_result[i] = new Array()
        for(let k=0;k<K;k++){
            matmul_result[i][k]=0
            for(let j=0;j<J;j++){
                matmul_result[i][k]+=a[i][j]*b[j][k]
            }
        }
    }
    return matmul_result
}
function zeros4D(a,b,c,d){
    let result = new Array()
    for(let i=0;i<a;i++){
        result[i] = new Array()
        for(let k=0;k<b;k++){
            result[i][k]=new Array()
            for(let j=0;j<c;j++){
                result[i][k][j]=new Array()
                for(let x=0;x<d;x++){
                    result[i][k][j][x]=0
                }
            }
        }
    }
    return result
}
function convolute(pic,kernals){
    let pic_z = pic.length
    let pic_r = pic[0].length
    let pic_c = pic[0][0].length
    let k_n = kernals.length
    let k_z = kernals[0].length
    let k_r = kernals[0][0].length
    let k_c = kernals[0][0][0].length
    let r_r = pic_r-k_r+1
    let r_c = pic_c-k_c+1
    if(pic_r<k_r || pic_r<k_c){
        console.log("warning: conv3D, matrix smaller than kernal")
    }
    if(pic_z!=k_z){
        console.log("warning: conv3D, box have different channel with kernal")
    }
    let conv_result = zeros4D(parseInt(k_n),parseInt(k_z),parseInt(r_r),parseInt(r_c))
    for(let j=0;j<k_r;j++){
        for(let k=0;k<k_c;k++){
            for(let i=0;i<pic_z;i++){
                for(let x=0;x<k_n;x++){
                    for(let y=0;y<pic_r-k_r+1;y++){
                        for(let z=0;z<pic_c-k_c+1;z++){
                            conv_result[x][i][y][z] += pic[i][j+y][k+z]*kernals[x][i][j][k]
                        }
                    }
                }
            }
        }
    }
    let conv_result2 = new Array()
    for(let i=0;i<k_n;i++){
        for(let k=1;k<k_z;k++){
            for(let j=0;j<pic_r-k_r+1;j++){
                for(let x=0;x<pic_c-k_c+1;x++){
                    conv_result[i][0][j][x]+=conv_result[i][k][j][x]
                }
            }
        }
        conv_result2[i]=conv_result[i][0]
    }
    return conv_result2
}
function pool(pic){
    let p_z = pic.length
    let p_r = pic[0].length
    let p_c = pic[0][0].length
    let r_r = Math.floor(p_r/2)+Math.ceil(p_r%2)
    let r_c = Math.floor(p_c/2)+Math.ceil(p_c%2)
    let pool_result=zeros4D(1,p_z,2*r_r,2*r_c)[0]
    let pool_result2=zeros4D(1,p_z,r_r,r_c)[0]

    for(let i=0;i<p_z;i++){
        for(let k=0;k<p_r;k++){
            for(let j=0;j<p_c;j++){
                pool_result[i][k][j]=pic[i][k][j]
            }
        }
    }
    for(let i=0;i<p_z;i++){
        for(let k=0;k<r_r;k++){
            for(let j=0;j<r_c;j++){
                pool_result2[i][k][j]=(pool_result[i][2*k][2*j]+pool_result[i][2*k+1][2*j]+pool_result[i][2*k][2*j+1]+pool_result[i][2*k+1][2*j+1])/4
            }
        }
    }
    return pool_result2
}
function sigmoid(x){
    return 1/(1+Math.exp(-x))
}
function relu(x){
    return x>=0? x : 0.01*x
}
function reshape(pic){
    let p_z = pic.length
    let p_r = pic[0].length
    let p_c = pic[0][0].length
    let reshape_result = [new Array()]
    for(let i=0;i<p_z;i++){
        for(let k=0;k<p_r;k++){
            for(let j=0;j<p_c;j++){
                reshape_result[0][i*p_r*p_c+k*p_c+j]=pic[i][k][j]
            }
        }
    }
    return reshape_result
}
function softmax(y){
    let sum = 0
    y = y.map((value)=>{
        return value.map((value)=>{
            let exp = Math.exp(value)
            sum += exp
            return exp
        })
    })
    y = y.map((value)=>{
        return value.map((value)=>{return value/sum})
    })
    return y
}
function select(y){
    let greatest = -100
    let id = -1
    for(let i=0;i<10;i++){
        if(y[0][i]>greatest){
            id = i
            greatest = y[0][i]
        }
    }
    return id
}