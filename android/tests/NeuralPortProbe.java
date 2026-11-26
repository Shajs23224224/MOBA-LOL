import org.aion.mobile.NeuralPort;
import java.io.*;
import java.util.*;
public class NeuralPortProbe {
    public static void main(String[] args)throws Exception{
        float[] v=new float[256];for(int i=0;i<256;i++)v[i]=(i%17-8)/8f;
        float[] result;
        try(InputStream in=new FileInputStream(args[0])){
            if(args.length>1)result=NeuralPort.expert(in,Integer.parseInt(args[1]),v);
            else {float[][] s=new float[16][32];for(int i=0;i<16;i++)Arrays.fill(s[i],i/16f);result=NeuralPort.load(in).fuse(v,s,Collections.singletonMap(0,v),new float[]{.2f,.3f,.5f,.1f});}
        }
        for(float f:result)System.out.println(f);
    }
}
