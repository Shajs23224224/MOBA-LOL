package org.aion.mobile;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

/** Portable Phase 8 ActorCritic inference, FP32. No action mapping or execution. */
public final class Policy {
    public static final int PARAMETERS = 34628;
    public static final int BYTES = 8 + PARAMETERS * 4 + 32;
    private final float[] w;
    private Policy(float[] weights) { w = weights; }
    public static Policy load(InputStream input) throws IOException {
        byte[] data = new byte[BYTES];
        int off=0;
        while (off<data.length) {
            int n=input.read(data,off,data.length-off);
            if (n<0) throw new IOException("Truncated policy");
            if (n==0) continue;
            off+=n;
        }
        if (input.read()!=-1) throw new IOException("Unexpected trailing data");
        byte[] hash;
        try { hash=MessageDigest.getInstance("SHA-256").digest(Arrays.copyOf(data,BYTES-32)); }
        catch (NoSuchAlgorithmException e) { throw new IOException(e); }
        if (!MessageDigest.isEqual(hash,Arrays.copyOfRange(data,BYTES-32,BYTES))) throw new IOException("Checksum mismatch");
        ByteBuffer b=ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        if (b.getInt()!=0x314e4941 || b.getInt()!=1) throw new IOException("Unsupported policy format");
        float[] weights=new float[PARAMETERS];
        for(int i=0;i<weights.length;i++) {
            weights[i]=b.getFloat();
            if (!Float.isFinite(weights[i]) || Math.abs(weights[i])>100) throw new IOException("Invalid weight");
        }
        return new Policy(weights);
    }
    /** Result layout: three logits, value, then 256 predicted features. */
    public float[] infer(float[] x) {
        if(x.length!=256) throw new IllegalArgumentException("256 features required");
        for(float v:x) if(!Float.isFinite(v)||Math.abs(v)>100) throw new IllegalArgumentException("Invalid feature");
        float[] a=linear(x,16448,4,256,-1);
        float[] h=linear(x,0,64,256,16384);
        float[] delta=linear(a,17472,64,4,-1);
        for(int i=0;i<64;i++) h[i]=(float)Math.tanh(h[i]+.25f*delta[i]);
        float[] logits=linear(h,17728,3,64,17920);
        float[] value=linear(h,17923,1,64,17987);
        float[] prediction=linear(h,17988,256,64,34372);
        float[] out=new float[260];
        System.arraycopy(logits,0,out,0,3); out[3]=value[0];
        System.arraycopy(prediction,0,out,4,256);
        return out;
    }
    private float[] linear(float[] x,int start,int rows,int cols,int bias) {
        float[] y=new float[rows];
        for(int r=0;r<rows;r++) {
            double s=bias<0?0:w[bias+r];
            for(int c=0;c<cols;c++) s+=(double)w[start+r*cols+c]*x[c];
            y[r]=(float)s;
        }
        return y;
    }
}
