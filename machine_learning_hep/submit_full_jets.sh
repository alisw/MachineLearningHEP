#!/bin/bash
DB_NAME="database_ml_parameters_D0pp_zg_0304"
DB_NAME_TMPL="database_ml_parameters_D0pp_zg_0304.yml"

Analysis_type=zg
declare -a Sys=(default fitting sideband cutvar powheg unfolding_prior)
declare -a NSys=(1 0 0 0 0 0) #1 9 21 17 9 1



for ((i=0; i<${#Sys[@]}; i++))
    do
	for ((j=1; j<=${NSys[$i]}; j++))
	do
	    if [ ${Sys[$i]} == "default" ]
	    then 
		rm -rf  /data/DerivedResultsJets/D0kAnywithJets/vAN-20200304_ROOT6-1/$Analysis_type/default
	    else
		rm -rf  /data/DerivedResultsJets/D0kAnywithJets/vAN-20200304_ROOT6-1/$Analysis_type/${Sys[$i]}/sys_$j
	    fi	    
	    db_name_curr="${DB_NAME}_${Sys[$i]}_$j.yaml"
	    cp data/JetAnalysis/$DB_NAME_TMPL data/JetAnalysis/systematics/$db_name_curr
	    sed -i "s/default/${Sys[$i]}/g" data/JetAnalysis/systematics/$db_name_curr
	    if [ ${Sys[$i]} == "default" ]
	    then
		sed -i "s/central_1/central/g" data/JetAnalysis/systematics/$db_name_curr
	    else
		sed -i "s/central_1/sys_$j/g" data/JetAnalysis/systematics/$db_name_curr
	    fi	
	    #less $db_name_curr
	    if [ ${Sys[$i]} == "fitting" ]
            then
		if [ $j == 1 ]
		then
		    sed -i "s/massmin_variable/1.76/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 2 ]
		then
		    sed -i "s/massmin_variable/1.74/g" data/JetAnalysis/systematics/$db_name_curr
		else
		    sed -i "s/massmin_variable/1.72/g" data/JetAnalysis/systematics/$db_name_curr
		fi
            else
		sed -i "s/massmin_variable/1.72/g" data/JetAnalysis/systematics/$db_name_curr
	    fi

	    if [ ${Sys[$i]} == "fitting" ] 
	    then
		if [ $j == 3 ]
		then
		    sed -i "s/massmax_variable/2.15/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 4 ]
		then
		    sed -i "s/massmax_variable/2.10/g" data/JetAnalysis/systematics/$db_name_curr
		else
		    sed -i "s/massmax_variable/2.05/g" data/JetAnalysis/systematics/$db_name_curr
		fi
	    else
		sed -i "s/massmax_variable/2.05/g" data/JetAnalysis/systematics/$db_name_curr
	    fi 
	    
	    if [ ${Sys[$i]} == "fitting" ] && [ $j == 5 ]
	    then
		sed -i "s/rebin_variable/12/g" data/JetAnalysis/systematics/$db_name_curr
	    else
		sed -i "s/rebin_variable/6/g" data/JetAnalysis/systematics/$db_name_curr
	    fi
	    
	    if [ ${Sys[$i]} == "fitting" ] && [ $j == 6 ]
	    then
		sed -i "s/fixedmean_variable/true/g" data/JetAnalysis/systematics/$db_name_curr
	    else
		sed -i "s/fixedmean_variable/false/g" data/JetAnalysis/systematics/$db_name_curr
	    fi
	    
	    if [ ${Sys[$i]} == "fitting" ] && [ $j == 7 ]
	    then
		sed -i "s/fixedsigma_variable/false/g" data/JetAnalysis/systematics/$db_name_curr
	    else
		sed -i "s/fixedsigma_variable/true/g" data/JetAnalysis/systematics/$db_name_curr
	    fi
	    
	    if [ ${Sys[$i]} == "fitting" ] && [ $j == 8 ]
	    then
		sed -i "s/bkgfunc_variable/2/g" data/JetAnalysis/systematics/$db_name_curr
	    else
		sed -i "s/bkgfunc_variable/0/g" data/JetAnalysis/systematics/$db_name_curr
	    fi
	    
	    if [ ${Sys[$i]} == "fitting" ] && [ $j == 9 ]
	    then
		sed -i "s/masspeak_variable/1.822/g" data/JetAnalysis/systematics/$db_name_curr
	    else
		sed -i "s/masspeak_variable/1.864/g" data/JetAnalysis/systematics/$db_name_curr
	    fi

	    if [ ${Sys[$i]} == "sideband" ]
	    then
		if [ $j == 1 ]
		then
		    sed -i "s/signalsigma_variable/1.6/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sigmascale_variable/0.890/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 2 ]
		then
		    sed -i "s/signalsigma_variable/1.7/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sigmascale_variable/0.911/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 3 ]
		then
		    sed -i "s/signalsigma_variable/1.8/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sigmascale_variable/0.928/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 4 ]
		then
		    sed -i "s/signalsigma_variable/1.9/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sigmascale_variable/0.943/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 5 ]
		then
		    sed -i "s/signalsigma_variable/2.1/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sigmascale_variable/0.964/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 6 ]
		then
		    sed -i "s/signalsigma_variable/2.2/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sigmascale_variable/0.972/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 7 ]
		then
		    sed -i "s/signalsigma_variable/2.3/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sigmascale_variable/0.979/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 8 ]
		then
		    sed -i "s/signalsigma_variable/2.4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sigmascale_variable/0.984/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 9 ]
		then
		    sed -i "s/signalsigma_variable/2.5/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sigmascale_variable/0.988/g" data/JetAnalysis/systematics/$db_name_curr
		else  
		    sed -i "s/signalsigma_variable/2.0/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sigmascale_variable/0.9545/g" data/JetAnalysis/systematics/$db_name_curr
		fi
	    else
		sed -i "s/signalsigma_variable/2.0/g" data/JetAnalysis/systematics/$db_name_curr
		sed -i "s/sigmascale_variable/0.9545/g" data/JetAnalysis/systematics/$db_name_curr	
	    fi

	    if [ ${Sys[$i]} == "sideband" ] 
	    then
		if [ $j == 10 ]
		then
		    sed -i "s/sidebandsigma1left_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/8/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/8/g" data/JetAnalysis/systematics/$db_name_curr
		    
		elif [ $j == 11 ]
		then
		    sed -i "s/sidebandsigma1left_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/6/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		    
		elif [ $j == 12 ]
		then
		    sed -i "s/sidebandsigma1left_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/6/g" data/JetAnalysis/systematics/$db_name_curr
		    
	        elif [ $j == 13 ]
		then
		    sed -i "s/sidebandsigma1left_variable/5/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/7/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		    
		elif [ $j == 14 ]
		then
		    sed -i "s/sidebandsigma1left_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/5/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/7/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 15 ]
		then
		    sed -i "s/sidebandsigma1left_variable/6/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/8/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 16 ]
		then
		    sed -i "s/sidebandsigma1left_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/7/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 17 ]
		then
		    sed -i "s/sidebandsigma1left_variable/5/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/8/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 18 ]
		then
		    sed -i "s/sidebandsigma1left_variable/5/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/8/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 19 ]
		then
		    sed -i "s/sidebandsigma1left_variable/5/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		elif [ $j == 20 ]
		then
		    sed -i "s/sidebandsigma1left_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/5/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		else
		    sed -i "s/sidebandsigma1left_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2left_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma1right_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		    sed -i "s/sidebandsigma2right_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		fi
	    else
		sed -i "s/sidebandsigma1left_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		sed -i "s/sidebandsigma2left_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
		sed -i "s/sidebandsigma1right_variable/4/g" data/JetAnalysis/systematics/$db_name_curr
		sed -i "s/sidebandsigma2right_variable/9/g" data/JetAnalysis/systematics/$db_name_curr
	    fi

	    if [ ${Sys[$i]} == "sideband" ] && [ $j == 21 ]
	    then
		sed -i "s/sidebandleftonly_variable/true/g" data/JetAnalysis/systematics/$db_name_curr
	    else
		sed -i "s/sidebandleftonly_variable/false/g" data/JetAnalysis/systematics/$db_name_curr
	    fi

	    if [ ${Sys[$i]} == "cutvar" ] 
	    then
		if [ $j == 1 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.82,0.80,0.72,0.70,0.50,0.50,0.50]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 2 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.83,0.81,0.73,0.71,0.51,0.51,0.51]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 3 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.84,0.82,0.74,0.72,0.52,0.52,0.52]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 4 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.85,0.83,0.75,0.73,0.53,0.53,0.53]/g" data/JetAnalysis/systematics/$db_name_curr
		fi
		
		if [ $j == 5 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.86,0.84,0.76,0.74,0.54,0.54,0.54]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 6 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.87,0.85,0.77,0.75,0.55,0.55,0.55]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 7 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.88,0.86,0.78,0.76,0.56,0.56,0.56]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 8 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.89,0.87,0.79,0.77,0.57,0.57,0.57]/g" data/JetAnalysis/systematics/$db_name_curr
		fi
		
		if [ $j == 9 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.90,0.88,0.80,0.78,0.58,0.58,0.58]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 10 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.91,0.89,0.81,0.79,0.59,0.59,0.59]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 11 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.93,0.91,0.83,0.81,0.61,0.61,0.61]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 12 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.94,0.92,0.84,0.82,0.62,0.62,0.62]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 13 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.95,0.93,0.85,0.83,0.63,0.63,0.63]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 14 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.96,0.94,0.86,0.84,0.64,0.64,0.64]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 15 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.97,0.95,0.87,0.85,0.65,0.65,0.65]/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 16 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.98,0.96,0.88,0.86,0.66,0.66,0.66]/g" data/JetAnalysis/systematics/$db_name_curr
		fi
		
		if [ $j == 17 ]
		then
		    sed -i "s/probcutoptimal_variable/[0.99,0.97,0.89,0.87,0.67,0.67,0.67]/g" data/JetAnalysis/systematics/$db_name_curr
		fi
	    else
		sed -i "s/probcutoptimal_variable/[0.92,0.90,0.82,0.80,0.60,0.60,0.60]/g" data/JetAnalysis/systematics/$db_name_curr
	    fi

	    if [ ${Sys[$i]} == "powheg" ] 
	    then
		if [ $j == 1 ]
		then
		    sed -i "s/powhegpathnonprompt_variable/F1_R05/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 2 ]
		then
		    sed -i "s/powhegpathnonprompt_variable/F05_R1/g" data/JetAnalysis/systematics/$db_name_curr
		fi
		
		if [ $j == 3 ]
		then
		    sed -i "s/powhegpathnonprompt_variable/F2_R1/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 4 ]
		then
		    sed -i "s/powhegpathnonprompt_variable/F1_R2/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 5 ]
		then
		    sed -i "s/powhegpathnonprompt_variable/F2_R2/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 6 ]
		then
		    sed -i "s/powhegpathnonprompt_variable/F05_R05/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 7 ]
		then
		    sed -i "s/powhegpathnonprompt_variable/Mhigh/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 8 ]
		then
		    sed -i "s/powhegpathnonprompt_variable/Mlow/g" data/JetAnalysis/systematics/$db_name_curr
		fi

		if [ $j == 9 ]
		then
		    sed -i "s/powhegpathnonprompt_variable/NoEvtGen/g" data/JetAnalysis/systematics/$db_name_curr
		fi
	    else
		sed -i "s/powhegpathnonprompt_variable/central/g" data/JetAnalysis/systematics/$db_name_curr
	    fi

	    if [ ${Sys[$i]} == "unfolding_prior" ] && [ $j == 1 ]
	    then
		sed -i "s/doprior_variable/true/g" data/JetAnalysis/systematics/$db_name_curr
	    else
		sed -i "s/doprior_variable/false/g" data/JetAnalysis/systematics/$db_name_curr
	    fi

	    
	    python do_entire_analysis.py -r submission/default_complete.yml -a jet_$Analysis_type -d data/JetAnalysis/systematics/$db_name_curr &
	done
   done
	
