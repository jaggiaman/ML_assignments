X = [3,1,0,4] #Input List
Y = [2,2,1,3] #Actual Output
N = len(X);#Length of the Input list
alpha = 0.25#Learning rate

# funtion to predict y values and to calculate new theta1 theta0 and error function
def predictYP(w1,w2):
    YP =[0] * N; #predicted Value
    error = 0;        #error
    a = 0; #theta0
    b = 0;#theta1
    for i in range(N):
        YP[i] = round(w1+w2*X[i],3)
        error += round(0.5*(pow((Y[i]-YP[i]),2)),4)#Error
        a += round((YP[i]-Y[i]),4)#partial derivative of J wrt theta0
        b += round((YP[i]-Y[i])*X[i],4)#partial derivative of J wrt theta1
    return  YP,error, a, b

#function to update wt
def Updt(wt):
    return round(alpha*(wt)/N,4)

def gradDesc(a,b):
    error = 0;
    count =0;
    print("no."+"\t\t"+ "predicted Y"+"\t   error"+"\t"+"Theta0"+"\t"+"Theta1")
    for i in range(5):
        YP, error, a_new, b_new = predictYP(a,b);
        print(str(i+1)+"\t"+ str(YP)+ "\t"+str(round(error,2))+"\t"+str(round(a,2))+"\t\t"+str(round(b,2)))
        a -= Updt(a_new);
        b -= Updt(b_new);
    return YP, round(error,2), round(a,2), round(b,2)

def run():
    Ypred, Error, theta0, theta1 = gradDesc(0,1)
    print("-------------------------------------------------------")
    print("Error : "+str(Error))
    print("theta0: "+ str(theta0))
    print("theta1: "+ str(theta1))
    print("Final Predicted values "+str(Ypred))



if __name__ == '__main__':
    run()