package org.aion.mobile;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.widget.*;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.*;

/** Local, manually labelled images; no capture, network, or game control. */
public final class LabActivity extends Activity {
    private static final ExecutorService WORK=Executors.newSingleThreadExecutor();
    private TextView status;
    private LinearLayout layout;
    private float[] features;
    private boolean busy;
    private Path store;
    @Override public void onCreate(Bundle state){
        super.onCreate(state);store=getFilesDir().toPath().resolve("mobile-lab-v1.bin");
        ScrollView scroll=new ScrollView(this);layout=new LinearLayout(this);layout.setOrientation(LinearLayout.VERTICAL);layout.setPadding(24,40,24,24);scroll.addView(layout);setContentView(scroll);
        TextView title=new TextView(this);title.setText("AION • Aprendizaje local");title.setTextSize(24);layout.addView(title);
        status=new TextView(this);status.setText("Perfil compacto independiente. Selecciona una imagen autorizada y asigna una etiqueta abstracta A, B o C. No son acciones del juego.");layout.addView(status);
        button("Seleccionar imagen",()->{Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT);i.addCategory(Intent.CATEGORY_OPENABLE);i.setType("image/*");startActivityForResult(i,1);});
        for(int n=0;n<3;n++){final int label=n;button("Registrar etiqueta "+(char)('A'+n),()->record(label));}
        button("Entrenar y validar localmente",()->task(()->{MobileLab lab=MobileLab.load(store);String result=lab.train();lab.save(store);return result;}));
        button("Consultar memoria y prediccion",()->{if(features==null){status.setText("Primero selecciona una imagen.");return;}float[] x=features.clone();task(()->{MobileLab lab=MobileLab.load(store);return "Registros: "+lab.size()+"\nProbabilidades no calibradas A/B/C: "+Arrays.toString(lab.probabilities(x))+"\nRecuerdo mas cercano (etiqueta, distancia): "+Arrays.toString(lab.recall(x));});});
    }
    private void button(String name,Runnable click){Button b=new Button(this);b.setText(name);b.setOnClickListener(v->{if(!busy)click.run();});layout.addView(b);}
    private void record(int label){if(features==null){status.setText("Primero selecciona una imagen.");return;}float[] x=features.clone();task(()->{MobileLab lab=MobileLab.load(store);boolean added=lab.add(x,label);lab.save(store);return (added?"Ejemplo guardado":"Ejemplo duplicado ignorado")+". Memoria: "+lab.size()+" / "+MobileLab.CAPACITY;});}
    private void task(Callable<String> callable){busy=true;status.setText("Procesando localmente...");WORK.execute(()->{try{finish(callable.call());}catch(Exception e){finish("Error: "+e.getMessage());}});}
    private void finish(String s){runOnUiThread(()->{if(!isDestroyed()){busy=false;status.setText(s);}});}
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);if(request!=1||result!=RESULT_OK||data==null||data.getData()==null||busy)return;Uri uri=data.getData();busy=true;features=null;status.setText("Leyendo imagen...");WORK.execute(()->{try{float[] x=decode(uri);runOnUiThread(()->{if(!isDestroyed()){features=x;busy=false;status.setText("256 caracteristicas del perfil movil RGB/contraste listas. Asigna una etiqueta o consulta la memoria.");}});}catch(Exception e){finish("Error: "+e.getMessage());}});}
    private float[] decode(Uri uri)throws Exception{
        BitmapFactory.Options o=new BitmapFactory.Options();o.inJustDecodeBounds=true;
        try(InputStream in=getContentResolver().openInputStream(uri)){BitmapFactory.decodeStream(in,null,o);}
        if(o.outWidth<1||o.outHeight<1||o.outWidth>32768||o.outHeight>32768)throw new IllegalArgumentException("Dimensiones invalidas");
        o.inJustDecodeBounds=false;o.inSampleSize=1;o.inPreferredConfig=Bitmap.Config.ARGB_8888;
        while((Math.max(o.outWidth,o.outHeight)+o.inSampleSize-1)/o.inSampleSize>512)o.inSampleSize*=2;
        Bitmap bmp;try(InputStream in=getContentResolver().openInputStream(uri)){bmp=BitmapFactory.decodeStream(in,null,o);}
        if(bmp==null)throw new IllegalArgumentException("Imagen no legible");
        try{int w=bmp.getWidth(),h=bmp.getHeight();if(w>512||h>512)throw new IllegalArgumentException("Imagen excede presupuesto");int[] pixels=new int[w*h];bmp.getPixels(pixels,0,w,0,0,w,h);return MobileLab.encode(pixels,w,h);}finally{bmp.recycle();}
    }
}
