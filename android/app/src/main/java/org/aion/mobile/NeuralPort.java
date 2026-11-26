package org.aion.mobile;
import java.io.*;
import java.security.*;
import java.util.*;

/** Exact Phase 2/9 dimensions; imported weights are diagnostic. */
public final class NeuralPort {
    public static final int FUSION_PARAMS=2163411, EXPERT_PARAMS=64061696;
    private final float[] weights;
    private NeuralPort(float[] w){weights=w;}
    private static final class Reader {
        final DataInputStream in; final DigestInputStream stream; final MessageDigest digest;
        Reader(InputStream input,int magic,int count,int id)throws IOException{
            try{digest=MessageDigest.getInstance("SHA-256");}catch(NoSuchAlgorithmException e){throw new IOException(e);}
            stream=new DigestInputStream(new BufferedInputStream(input,65536),digest);in=new DataInputStream(stream);
            if(in.readInt()!=magic||in.readInt()!=count||in.readInt()!=id)throw new IOException("wrong neural header");
        }
        float value()throws IOException{float v=in.readFloat();if(!Float.isFinite(v)||Math.abs(v)>100)throw new IOException("invalid weight");return v;}
        float[] read(int n)throws IOException{float[] a=new float[n];for(int i=0;i<n;i++)a[i]=value();return a;}
        void end()throws IOException{byte[] hash=digest.digest();stream.on(false);byte[] actual=new byte[32];in.readFully(actual);if(!MessageDigest.isEqual(hash,actual)||in.read()!=-1)throw new IOException("checksum mismatch");}
    }
    public static NeuralPort load(InputStream in)throws IOException{Reader r=new Reader(in,0x41464e31,FUSION_PARAMS,0);float[] w=r.read(FUSION_PARAMS);r.end();return new NeuralPort(w);}
    private static void check(float[] x,int n){if(x.length!=n)throw new IllegalArgumentException("invalid shape");for(float v:x)if(!Float.isFinite(v)||Math.abs(v)>10000)throw new IllegalArgumentException("invalid input");}
    private static float gelu(float x){double z=x/Math.sqrt(2),a=Math.abs(z),t=1/(1+.3275911*a);double erf=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-a*a);return (float)(.5*x*(1+Math.copySign(erf,z)));}
    private static float[] join(float[]... arrays){int n=0;for(float[] a:arrays)n+=a.length;float[] y=new float[n];int i=0;for(float[] a:arrays){System.arraycopy(a,0,y,i,a.length);i+=a.length;}return y;}
    private float[] linear(float[] x,int out,int[] pos){int start=pos[0],bias=start+x.length*out;float[] y=new float[out];for(int i=0;i<out;i++){double s=weights[bias+i];for(int j=0;j<x.length;j++)s+=(double)weights[start+i*x.length+j]*x[j];y[i]=(float)s;}pos[0]=bias+out;return y;}
    private float[] mlp(float[] x,int hidden,int out,int[] pos){float[] h=linear(x,hidden,pos);for(int i=0;i<h.length;i++)h[i]=gelu(h[i]);return linear(h,out,pos);}
    /** Output: 16 x (64 latent + 4 estimates), 16 F01 logits, 3 F02 logits. */
    public float[] fuse(float[] visual,float[][] summaries,Map<Integer,float[]> experts,float[] actor){
        check(visual,256);check(actor,4);if(summaries.length!=16||experts.size()>2)throw new IllegalArgumentException("invalid subsystem count");
        for(int id:experts.keySet()){if(id<0||id>=16)throw new IllegalArgumentException("invalid subsystem");check(experts.get(id),256);}
        float[] global=new float[1092];int[] pos={0};
        for(int i=0;i<16;i++){check(summaries[i],32);boolean present=experts.containsKey(i);float[] x=join(visual,experts.getOrDefault(i,new float[256]),summaries[i],new float[]{present?1:0});float[] latent=mlp(x,128,64,pos);float[] raw=mlp(join(x,latent),64,4,pos);raw[0]=(float)Math.tanh(raw[0]);for(int j=1;j<4;j++)raw[j]=(float)(1/(1+Math.exp(-raw[j])));System.arraycopy(latent,0,global,i*68,64);System.arraycopy(raw,0,global,i*68+64,4);}
        System.arraycopy(actor,0,global,1088,4);float[] strategic=mlp(global,128,16,pos),policy=mlp(join(global,strategic),128,3,pos);if(pos[0]!=weights.length)throw new IllegalStateException("weight count");float[] out=join(Arrays.copyOf(global,1088),strategic,policy);check(out,1107);return out;
    }
    private static float[] streamedLinear(Reader r,float[] x,int n)throws IOException{float[] y=new float[n];for(int i=0;i<n;i++){double s=0;for(float v:x)s+=(double)r.value()*v;y[i]=(float)s;}for(int i=0;i<n;i++)y[i]+=r.value();return y;}
    /** One explicitly selected expert, no full dense weight array; hash checked before output. */
    public static float[] expert(InputStream input,int id,float[] visual)throws IOException{
        if(id<1||id>824)throw new IllegalArgumentException("expert ID outside catalog");check(visual,256);Reader r=new Reader(input,0x41454e31,EXPERT_PARAMS,id);float[] x=visual;
        for(int layer=0;layer<16;layer++){float[] y=streamedLinear(r,x,2048),scale=r.read(2048),bias=r.read(2048);double mean=0,var=0;for(float v:y)mean+=v;mean/=2048;for(float v:y)var+=(v-mean)*(v-mean);double denom=Math.sqrt(var/2048+1e-5);for(int i=0;i<2048;i++){y[i]=gelu((float)((y[i]-mean)/denom*scale[i]+bias[i]));if(layer>0)y[i]+=x[i];}x=y;}
        float[] output=streamedLinear(r,x,256);r.end();check(output,256);return output;
    }
}
