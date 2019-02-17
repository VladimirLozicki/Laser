import framework.Constants;
import framework.KernelBuilder;
import framework.ResonatorBuilder;
import framework.entities.Resonator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jocl.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;
import java.util.stream.IntStream;
import java.util.stream.*;
import java.util.Random;
import java.lang.*;
import java.io.*;



import static org.jocl.CL.*;

public class HostPart {

    private static final Logger logger = LogManager.getLogger();
    
    // размер коммуникатора
    private static final int maxGlobalWorkSize =2;
    private static final int xpoints = 1000;
    private static final int save_freq =1; // частота сохранения результата
    private static double tpoints = 40000.0;
    private static double dt = 0.001;
    private static double Rleft = 1.0;
    private static double tmax=4700.0;
    private static double Rright=1.0;
    private static double whole_lenght=2.0;
    private static double dx = whole_lenght / (xpoints-1);

    // число точек на процесс
    private static double local_n = xpoints/maxGlobalWorkSize;
    

    private static cl_program program;
    private static cl_kernel kernel;
    private static cl_context context;
    private static cl_command_queue commandQueue;
    private static cl_mem memObjects[] = new cl_mem[3];

    
    /**
     * Main part of the Host Part.
     *
     * @param args Not used.
     */

    public static void main(String args[]) {
        // Create input- and output data
        double[] U_plus = new double[xpoints];


        //массив произедения элементов


        double[] U_minus=new double[xpoints];
        double[] D = new double[xpoints];
        // создать два буффера и использовать их как начальные и конечные значения
         
         // round целочисленное округление
         //int T=(int)Math.round(tmax/dt);
         // пустить ступеньку как полуширина гауса  высота 1 
          //double amplitude = 30000; //берём максимальную возможную амплитуду
	  //double freq_Hz = 50; 
          //double S_RATE=1;
          IntStream.range(0, xpoints).forEach((i) -> {
          U_plus[i]=1/(0.1*Math.sqrt(2*Math.PI)*Math.exp((i*dx-0.5)*(i*dx-0.5)/(2*0.01))+(Math.random()*1e-12));
          U_minus[i]=(Math.random()*1e-12);
          D[i]=Math.random()*1e-5;
        });
       
        try (final FileWriter writer = new FileWriter("/Users/vladimirlozickiy/Desktop/Laser/src/main/java/kernels/temp.csv", false))
        {
            for (int i = 0; i < U_plus.length; i++)
            {
                final String s = Double.toString(U_plus[i]);
                writer.write(s);
                writer.write(System.lineSeparator());
            }
        }
        catch(IOException e) {
            System.out.println(e.getMessage());
        }
        
        double initialWave=0;
        for (int p=0; p<xpoints;p++)
        { 
            initialWave=initialWave+U_plus[p];
           
            
        }
       // Pointer srcA = Pointer.to(temporaryRight);
      //  Pointer srcB = Pointer.to(temporaryLeft);
        // right правая волна
        //tmp лева] волна
        Pointer u_plus = Pointer.to(U_plus);
        Pointer u_minus = Pointer.to(U_minus);
        Pointer d = Pointer.to(D);
        initialize(u_plus, u_minus, d);
       // initialize(right, left);

        // Set the arguments for the kernel
        // добавить новый элемент(счетчик элементов массива)
        
        
        //Задаем аргументы ядра
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memObjects[0]));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memObjects[1]));
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memObjects[2]));

        // Set the work-item dimensions
        long global_work_size[] = new long[]{maxGlobalWorkSize};
        long local_work_size[] = new long[]{1};

        // Execute the kernel
       
        /*
          выполнение ядра
        */ 
       
   

   for(int t=0; t<tmax; t+=xpoints*dt )
   {
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);
        // Read the output data
        // чтение из буфер
        
        clEnqueueReadBuffer(commandQueue, memObjects[0], CL_TRUE, 0, xpoints * Sizeof.cl_double, u_plus, 0, null, null);
        clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE, 0, xpoints * Sizeof.cl_double, u_minus, 0, null, null);
        clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, xpoints * Sizeof.cl_double, d, 0, null, null);
       // clEnqueueReadBuffer(commandQueue, memObjects[3], CL_TRUE, 0, xpoints * Sizeof.cl_double, left, 0, null, null);
           // запись в файл правой волны
//         try (final FileWriter writer = new FileWriter("/Users/vladimir/Desktop/LaserDynamicsModelling/src/main/U_plus.csv", false))
//        {
//             for (int i =0; i < U_plus.length; i++)
//            {
//                final String s = Double.toString(U_plus[i]);
//                writer.write(s);
//                writer.write(System.lineSeparator());
//                //U_plus[i]=U_plus[i]+(Math.random()*1e-12);
//            }
//          // System.out.println("u_plus.lenght" + u_plus.length);
//        }
//        catch(IOException e) {
//            System.out.println(e.getMessage());
//         }

             try (final FileWriter writer = new FileWriter("/Users/vladimirlozickiy/Desktop/Laser/src/main/java/kernels/U_minus.csv", false))
        {
             for (int i=U_minus.length-1; i>=0; i--)
            {
                final String s = Double.toString(U_minus[i]);
                writer.write(s);
                writer.write(System.lineSeparator());
                //U_plus[i]=U_plus[i]+(Math.random()*1e-12);
            }
          // System.out.println("u_plus.lenght" + u_plus.length);
        }
        catch(IOException e) {
            System.out.println(e.getMessage());
         }
             
             
           try (final FileWriter writer = new FileWriter("/Users/vladimirlozickiy/Desktop/Laser/src/main/java/kernels/U_plus.csv", false))
        {
            for (int i = 0; i < U_plus.length; i++)
            {
                final String s = Double.toString(U_plus[i]);
                writer.write(s);
                writer.write(System.lineSeparator());
               // U_minus[i]=U_minus[i]+(Math.random()*1e-12);
            }
        }
        catch(IOException e) {
            System.out.println(e.getMessage());
        }

       try (final FileWriter writer = new FileWriter("/Users/vladimirlozickiy/Desktop/Laser/src/main/java/kernels/D.csv", false))
       {
           for (int i = 0; i < D.length; i++)
           {
               final String s = Double.toString(D[i]);
               writer.write(s);
               writer.write(System.lineSeparator());
           }
       }
       catch(IOException e) {
           System.out.println(e.getMessage());
       }

   }
       /*
         //проверяем волну после одного шага по времени
         */
        double oneStep=0;
        double difference;
        for (int p=0; p<xpoints;p++)
        { 
            oneStep=oneStep+U_plus[p];
        }
          difference=initialWave-oneStep;
         System.out.println("волна начальная: " + initialWave);
         System.out.println("волна с шагом dt: " + oneStep);
         System.out.println("разность : " + difference);

         
        //output the results to the console
        
         //IntStream.range(0, U_plus.length).mapToDouble(i -> U_plus[i]).forEach(System.out::println);
        //IntStream.range(0, u_minus.length).mapToDouble(i -> u_minus[i]).forEach(System.out::println);
       
      
         shutdown();
    } 

    /**
     * initializes all main parts of the program as kernel, program, command queue, context etc.
     *
     * @param pointerA pointer to the first input data array.
     * @param pointerB pointer to the second input data array.
     */
    
    //инициализация основных частей программы
    //////
    private static void initialize(final Pointer pointerA, final Pointer pointerB, final Pointer pointerD) {
        // The platform, device type and device number that will be used
        	
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];
        
        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];
      //  System.out.println("Device: " + device.getType());

        // Create a context for the selected device
        // создаем контекст
        context = clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);

        // Create a command-queue for the selected device
        // функция создания очереди команд
        commandQueue = clCreateCommandQueue(context, device, 0, null);

        // Allocate the memory objects for the input- and output data
        // создаем буфер
        memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double * xpoints, pointerA, null);
        memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double * xpoints, pointerB, null);
        memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_double * xpoints, pointerD, null);
        String programSource =
               readFile("/Users/vladimirlozickiy/Desktop/Laser/src/main/java/kernels/LaserDynamics.cl")
                       .replace( "{{xpoints}}", String.format(Locale.US,"%d",xpoints) )
                       .replace("{{save_freq}}", String.format(Locale.US,"%d", save_freq) )
                       .replace("{{maxGlobalWorkSize}}", String.format(Locale.US,"%d", maxGlobalWorkSize) )
                       .replace("{{R_left}}", String.format(Locale.US,"%f",Rleft) )
                       .replace("{{R_right}}", String.format(Locale.US,"%f",Rright) )
                       .replace("{{dt}}", String.format(Locale.US,"%f",dt) )
                       .replace("{{dx}}", String.format(Locale.US,"%f",dx) )
                       .replace("{{local_n}}", String.format(Locale.US,"%f",local_n) )
               ;

           // equals 
           // метод equals сравнение строк
       // String programSource;
        /*if (Constants.kernelCreator.equals(Constants.KernelCreator.CODE.name())) {
            
           // Create kernel file from the source code
           // создаем файл ядра из исходного кода
            
           Resonator resonator = new ResonatorBuilder().buildResonator();
            programSource = new KernelBuilder(resonator).buildKernel();
        } else {
           // Create the program from the existing kernel file
            //  Создает программу из существующего файла ядра
            int lalala=400;
            programSource = readFile(Constants.clFileName).replace("{{lalala}}", String.format("%d",lalala) ); 
        }*/




        program = clCreateProgramWithSource(context, 1, new String[]{programSource}, null, null);

        // Build the program
        
        
        try 
        {
             clBuildProgram(program, 0, null, null, null, null);
         }
         catch (org.jocl.CLException e)
         {
             System.out.println("Ошибка " + e.getMessage()+ "...");
             e.printStackTrace();
         }
       
        // Create the kernel
        kernel = clCreateKernel(program, Constants.kernelName, null);
    }

    /**
     * Clearing memory that points to objects.
     *
     * @param objects objects to release.
     */
    private static void releaseObjects(final cl_mem[] objects) {
        Arrays.stream(objects).forEach(CL::clReleaseMemObject);
    }

    /**
     * Read the contents of the file with the given name, and return it as a string.
     *
     * @param fileName The name of the file to read.
     * @return The contents of the file.
     */
    
    private static String readFile(final String fileName) {
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            StringBuilder sb = new StringBuilder();
            String line = null;
            while (true) {
                line = br.readLine();
                if (line == null) {
                    break;
                }
                sb.append(line).append("\n");
            }
            return sb.toString();
        } catch (IOException e) {
            logger.fatal(String.format("Can't convert %s file to string : ", fileName) + e);
            return "";
        }
    }
    /**
     * Release created kernel, program, command queue and context.
     */
    private static void shutdown() {
        // Release kernel, program, and memory objects
        releaseObjects(memObjects);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }
}




