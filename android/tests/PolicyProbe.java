import org.aion.mobile.Policy;
import java.io.FileInputStream;
public class PolicyProbe {
    public static void main(String[] args) throws Exception {
        try (FileInputStream stream=new FileInputStream(args[0])) {
            Policy p=Policy.load(stream);
            float[] x=new float[256];
            for(int i=0;i<x.length;i++) x[i]=(i%17-8)/8f;
            float[] y=p.infer(x);
            for(float v:y) System.out.println(v);
        }
    }
}
