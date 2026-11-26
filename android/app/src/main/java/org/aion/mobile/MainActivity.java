package org.aion.mobile;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Intent;
import android.os.Bundle;
import android.net.Uri;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.view.View;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONArray;

/** Read-only local research diagnostics; no overlay, accessibility or input injection. */
public class MainActivity extends Activity {
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private TextView status;
    private Button start, modelButton, imageButton, vectorButton;
    private Policy model;
    private float[] features=fixture();
    private String source="vector sintetico";
    private boolean busy;
    private static float[] fixture() {
        float[] x=new float[256]; for(int i=0;i<x.length;i++) x[i]=(i%17-8)/8f; return x;
    }
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll=new ScrollView(this);
        LinearLayout layout=new LinearLayout(this); layout.setOrientation(LinearLayout.VERTICAL); layout.setPadding(24,32,24,32);
        TextView title=new TextView(this); title.setText("AION | Android 0.9.1"); title.setTextSize(26); layout.addView(title);
        ActivityManager manager=(ActivityManager)getSystemService(ACTIVITY_SERVICE);
        TextView detail=new TextView(this); detail.setText("Diagnostico local • sin control de juego\nHeap: "+manager.getMemoryClass()+" MB • CPU: "+Runtime.getRuntime().availableProcessors()+" hilos\n858 estructuras en el nucleo de investigacion; 34.628 parametros en el runtime movil auxiliar."); layout.addView(detail);
        modelButton=button(layout,"Importar pesos .aion",v->pick(1,"application/octet-stream"));
        vectorButton=button(layout,"Importar vector JSON (256)",v->pick(2,"application/json"));
        imageButton=button(layout,"Examinar imagen local",v->pick(3,"image/*"));
        start=button(layout,"INICIAR DIAGNOSTICO",v->runPolicy());
        status=new TextView(this); status.setTextIsSelectable(true); status.setPadding(0,24,0,0); layout.addView(status);
        status.setText("Importa un modelo .aion exportado por AION. El vector sintetico es solo para diagnostico. No hay pesos entrenados incluidos.\nUltimo resultado: "+getPreferences(MODE_PRIVATE).getString("last","ninguno"));
        scroll.addView(layout); setContentView(scroll); updateButtons();
    }
    private Button button(LinearLayout layout,String label,View.OnClickListener listener) {
        Button b=new Button(this); b.setText(label); b.setOnClickListener(listener); layout.addView(b); return b;
    }
    private void updateButtons() {
        start.setEnabled(!busy&&model!=null); modelButton.setEnabled(!busy); vectorButton.setEnabled(!busy); imageButton.setEnabled(!busy);
    }
    private void pick(int request,String type) {
        Intent i=new Intent(Intent.ACTION_OPEN_DOCUMENT); i.addCategory(Intent.CATEGORY_OPENABLE); i.setType(type); startActivityForResult(i,request);
    }
    @Override protected void onActivityResult(int request,int result,Intent data) {
        super.onActivityResult(request,result,data);
        if(result!=RESULT_OK||data==null||data.getData()==null||busy) return;
        Uri uri=data.getData(); busy=true; updateButtons();
        worker.execute(()-> {
            try {
                if(request==1) {
                    Policy next;
                    try(InputStream input=getContentResolver().openInputStream(uri)) { if(input==null) throw new IllegalArgumentException("Archivo no disponible"); next=Policy.load(input); }
                    runOnUiThread(()->{ model=next; finishWork("Modelo cargado. Integridad SHA-256 verificada; procedencia y entrenamiento no certificados."); });
                } else if(request==2) {
                    float[] x=readVector(uri);
                    runOnUiThread(()->{ features=x; source="vector importado (compatibilidad no certificada)"; finishWork("Vector cargado. Verifica que corresponda a la representacion del checkpoint."); });
                } else if(request==3) {
                    String report=inspectImage(uri);
                    runOnUiThread(()->finishWork(report));
                } else runOnUiThread(()->finishWork("Solicitud desconocida"));
            } catch(Exception e) { runOnUiThread(()->finishWork("Error: "+e.getMessage())); }
        });
    }
    private float[] readVector(Uri uri) throws Exception {
        try(InputStream in=getContentResolver().openInputStream(uri); ByteArrayOutputStream out=new ByteArrayOutputStream()) {
            if(in==null) throw new IllegalArgumentException("Archivo no disponible");
            byte[] buf=new byte[4096]; int n;
            while((n=in.read(buf))!=-1) { if(out.size()+n>65536) throw new IllegalArgumentException("JSON demasiado grande"); out.write(buf,0,n); }
            JSONArray array=new JSONArray(out.toString("UTF-8"));
            if(array.length()!=256) throw new IllegalArgumentException("Se requieren 256 valores");
            float[] x=new float[256]; for(int j=0;j<256;j++) { x[j]=(float)array.getDouble(j); if(!Float.isFinite(x[j])||Math.abs(x[j])>100) throw new IllegalArgumentException("Valor no valido"); } return x;
        }
    }
    private String inspectImage(Uri uri) throws Exception {
        BitmapFactory.Options options=new BitmapFactory.Options(); options.inJustDecodeBounds=true;
        try(InputStream in=getContentResolver().openInputStream(uri)) { BitmapFactory.decodeStream(in,null,options); }
        int width=options.outWidth,height=options.outHeight;
        if(width<=0||height<=0||width>32768||height>32768) throw new IllegalArgumentException("Dimensiones invalidas");
        options.inJustDecodeBounds=false; options.inSampleSize=1;
        while(Math.max(width,height)/options.inSampleSize>512) options.inSampleSize*=2;
        Bitmap bmp;
        try(InputStream in=getContentResolver().openInputStream(uri)) { bmp=BitmapFactory.decodeStream(in,null,options); }
        if(bmp==null) throw new IllegalArgumentException("Imagen no legible");
        try {
            long r=0,g=0,b=0; int count=bmp.getWidth()*bmp.getHeight();
            for(int y=0;y<bmp.getHeight();y++) for(int x=0;x<bmp.getWidth();x++) { int p=bmp.getPixel(x,y); r+=(p>>16)&255; g+=(p>>8)&255; b+=p&255; }
            return "Imagen: "+width+" x "+height+"\nRGB promedio aproximado: "+r/count+", "+g/count+", "+b/count+"\nDiagnostico visual separado. No equivale al vector OpenCV/OCR ni cambia el vector de inferencia.";
        } finally { bmp.recycle(); }
    }
    private void runPolicy() {
        if(model==null||busy) return; busy=true; updateButtons();
        Policy current=model; float[] x=features.clone(); String origin=source;
        worker.execute(()-> {
            try {
                long startTime=System.nanoTime(); float[] y=current.infer(x); double ms=(System.nanoTime()-startTime)/1e6;
                String report=String.format(java.util.Locale.ROOT,"%s\nLogits abstractos: %.5f, %.5f, %.5f\nValor: %.5f\nPrediccion: 256 valores\nInferencia: %.2f ms\nSolo diagnostico; no acciones.",origin,y[0],y[1],y[2],y[3],ms);
                runOnUiThread(()-> { if(!isDestroyed()){getPreferences(MODE_PRIVATE).edit().putString("last",report).apply(); finishWork(report);} });
            } catch(Exception e) { runOnUiThread(()->finishWork("Error: "+e.getMessage())); }
        });
    }
    private void finishWork(String message) { if(isDestroyed()) return; busy=false; status.setText(message); updateButtons(); }
    @Override protected void onDestroy() { worker.shutdownNow(); super.onDestroy(); }
}
