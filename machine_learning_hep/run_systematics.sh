#!/bin/bash
DB_NAME="database_ml_parameters_LcpK0spp_systematics"
DB_NAME_TMPL="database_ml_parameters_LcpK0spp_systematics.yml"

#declare -a Sys=(fitting cutvar powheg)
#declare -a NSys=(2 1 1)
declare -a Sys=(fitting sideband cutvar powheg unfolding_prior)
declare -a NSys=(6 10 9 9 1) #6 10 9 9 1



for ((i=0; i<${#Sys[@]}; i++))
    do
	for ((j=1; j<=${NSys[$i]}; j++))
	do
	    db_name_curr="${DB_NAME}_${Sys[$i]}_$j.yaml"
	    cp data/$DB_NAME_TMPL data/$db_name_curr
	    sed -i "s/default/${Sys[$i]}/g" data/$db_name_curr
	    sed -i "s/sys_1/sys_$j/g" data/$db_name_curr
	    #less $db_name_curr
	    if [ ${Sys[$i]} == "Fitting" ] && [ $j == 1 ]
	    then
		sed -i "s/massmin_variable/2.18/g" data/$db_name_curr
	    else
		sed -i "s/massmin_variable/2.14/g" data/$db_name_curr
	    fi
	    
	    if [ ${Sys[$i]} == "Fitting" ] && [ $j == 2 ]
	    then
		sed -i "s/massmax_variable/2.476/g" data/$db_name_curr
	    else
		sed -i "s/massmax_variable/2.436/g" data/$db_name_curr
	    fi
	    
	    if [ ${Sys[$i]} == "Fitting" ] && [ $j == 3 ]
	    then
		sed -i "s/rebin_variable/12/g" data/$db_name_curr
	    else
		sed -i "s/rebin_variable/6/g" data/$db_name_curr
	    fi
	    
	    if [ ${Sys[$i]} == "Fitting" ] && [ $j == 4 ]
	    then
		sed -i "s/FixedMean_variable/true/g" data/$db_name_curr
	    else
		sed -i "s/FixedMean_variable/false/g" data/$db_name_curr
	    fi
	    
	    if [ ${Sys[$i]} == "Fitting" ] && [ $j == 5 ]
	    then
		sed -i "s/Fixed_sigma_variable/false/g" data/$db_name_curr
	    else
		sed -i "s/Fixed_sigma_variable/true/g" data/$db_name_curr
	    fi
	    if [ ${Sys[$i]} == "Fitting" ] && [ $j == 6 ]
	    then
		sed -i "s/bkgfunc_variable/kExpo/g" data/$db_name_curr
	    else
		sed -i "s/bkgfunc_variable/Pol2/g" data/$db_name_curr
	    fi
	    
	    
	    if [ ${Sys[$i]} == "sideband" ] && [ $j == 1 ]
	    then
		sed -i "s/signal_sigma_variable/1.5/g" data/$db_name_curr
		sed -i "s/sigma_scale_variable/0.866/g" data/$db_name_curr
	    else
		sed -i "s/signal_sigma_variable/2/g" data/$db_name_curr
		sed -i "s/sigma_scale_variable/0.9545/g" data/$db_name_curr
	    fi

	    if [ ${Sys[$i]} == "sideband" ] 
	    then
		if [ $j == 2 ]
		then
		    sed -i "s/sideband_sigma_1_left_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_left_variable/8/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_1_right_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_right_variable/8/g" data/$db_name_curr
		    
		elif [ $j == 3 ]
		then
		    sed -i "s/sideband_sigma_1_left_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_left_variable/6/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_1_right_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_right_variable/9/g" data/$db_name_curr
		    
		elif [ $j == 4 ]
		then
		    sed -i "s/sideband_sigma_1_left_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_left_variable/9/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_1_right_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_right_variable/6/g" data/$db_name_curr
		    
	        elif [ $j == 5 ]
		then
		    sed -i "s/sideband_sigma_1_left_variable/5/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_left_variable/7/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_1_right_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_right_variable/9/g" data/$db_name_curr
		    
		elif [ $j == 6 ]
		then
		    sed -i "s/sideband_sigma_1_left_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_left_variable/9/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_1_right_variable/5/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_right_variable/7/g" data/$db_name_curr
		elif [ $j == 7 ]
		then
		    sed -i "s/sideband_sigma_1_left_variable/6/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_left_variable/8/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_1_right_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_right_variable/9/g" data/$db_name_curr
		elif [ $j == 8 ]
		then
		    sed -i "s/sideband_sigma_1_left_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_left_variable/9/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_1_right_variable/6/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_right_variable/8/g" data/$db_name_curr
		elif [ $j == 9 ]
		then
		    sed -i "s/sideband_sigma_1_left_variable/7/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_left_variable/9/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_1_right_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_right_variable/9/g" data/$db_name_curr
		elif [ $j == 10 ]
		then
		    sed -i "s/sideband_sigma_1_left_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_left_variable/9/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_1_right_variable/7/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_right_variable/9/g" data/$db_name_curr
		else
		    sed -i "s/sideband_sigma_1_left_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_left_variable/9/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_1_right_variable/4/g" data/$db_name_curr
		    sed -i "s/sideband_sigma_2_right_variable/9/g" data/$db_name_curr
		fi
	    else
		sed -i "s/sideband_sigma_1_left_variable/4/g" data/$db_name_curr
		sed -i "s/sideband_sigma_2_left_variable/9/g" data/$db_name_curr
		sed -i "s/sideband_sigma_1_right_variable/4/g" data/$db_name_curr
		sed -i "s/sideband_sigma_2_right_variable/9/g" data/$db_name_curr
	    fi


	    if [ ${Sys[$i]} == "cutvar" ] 
	    then
		if [ $j == 1 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.3,0.3,0.3,0.3,0.3]/g" data/$db_name_curr
		fi

		if [ $j == 2 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.325,0.325,0.325,0.31,0.31]/g" data/$db_name_curr
		fi
		
		if [ $j == 3 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.35,0.35,0.35,0.32,0.32]/g" data/$db_name_curr
		fi

		if [ $j == 4 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.375,0.375,0.375,0.33,0.33]/g" data/$db_name_curr
		fi
		
		if [ $j == 5 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.4,0.4,0.4,0.34,0.34]/g" data/$db_name_curr
		fi

		if [ $j == 6 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.425,0.425,0.425,0.35,0.35]/g" data/$db_name_curr
		fi

		if [ $j == 7 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.45,0.45,0.45,0.36,0.36]/g" data/$db_name_curr
		fi

		if [ $j == 8 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.475,0.475,0.475,0.37,0.37]/g" data/$db_name_curr
		fi
		
		if [ $j == 9 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.5,0.5,0.5,0.38,0.38]/g" data/$db_name_curr
		fi
	    else
		sed -i "s/probcutoptimal_variable/[0.4,0.4,0.4,0.3,0.3]/g" data/$db_name_curr
	    fi
	   


	    if [ ${Sys[$i]} == "powheg" ] 
	    then
		if [ $j == 1 ]
		then
		    sed -i "s/powheg_path_nonprompt_variable/F1_R05/g" data/$db_name_curr
		fi

		if [ $j == 2 ]
		then
		    sed -i "s/powheg_path_nonprompt_variable/F05_R1/g" data/$db_name_curr
		fi
		
		if [ $j == 3 ]
		then
		    sed -i "s/powheg_path_nonprompt_variable/F2_R1/g" data/$db_name_curr
		fi

		if [ $j == 4 ]
		then
		    sed -i "s/powheg_path_nonprompt_variable/F1_R2/g" data/$db_name_curr
		fi

		if [ $j == 5 ]
		then
		    sed -i "s/powheg_path_nonprompt_variable/F2_R2/g" data/$db_name_curr
		fi

		if [ $j == 6 ]
		then
		    sed -i "s/powheg_path_nonprompt_variable/F05_R05/g" data/$db_name_curr
		fi

		if [ $j == 7 ]
		then
		    sed -i "s/powheg_path_nonprompt_variable/Mhigh/g" data/$db_name_curr
		fi

		if [ $j == 8 ]
		then
		    sed -i "s/powheg_path_nonprompt_variable/Mlow/g" data/$db_name_curr
		fi

		if [ $j == 9 ]
		then
		    sed -i "s/powheg_path_nonprompt_variable/NoEvtGen/g" data/$db_name_curr
		fi
	    else
		sed -i "s/powheg_path_nonprompt_variable/central/g" data/$db_name_curr
	    fi

	    if [ ${Sys[$i]} == "unfolding_prior" ] && [ $j == 1 ]
	    then
		sed -i "s/doprior_variable/true/g" data/$db_name_curr
	    else
		sed -i "s/doprior_variable/false/g" data/$db_name_curr
	    fi

	    
	    python do_entire_analysis.py -d data/$db_name_curr
	done
   done
	
