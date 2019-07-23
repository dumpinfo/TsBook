$(document).ready(function() {
    initPage();
    if(!verifyAndSetId()) {
		return;
	}
	
	$("#underwaylink").bind("click",function() {
	    menuLink(1);
	});
	$("#newinvitationlink").bind("click",function() {
	    menuLink(2);
	});
	$("#finishedlink").bind("click",function() {
	    menuLink(3);
	});
	$("#unfinishedlink").bind("click",function() {
	    menuLink(4);
	});
	
	//alert(localStorage.getItem("appointtype"));
	if(localStorage.getItem("appointtype") != null) {
		var appointtype = localStorage.getItem("appointtype");
		menuLink(appointtype);
	}
	
	ajaxGet("getPatientAppointListForWechat&userId=" + userId, onGetPatientAppointListForWechatOk, onGetPatientAppointListForWechatError);	
	
});

function onGetPatientAppointListForWechatOk(json) {
    if(json.status == "Ok") {
	    //正在进行--预约
	    var appointlist = json.appointsGetAppointList;
		onGetAppointListOk(appointlist);
		
		//正在进行--邀请
		var invitelist = json.appointsGetInvitationList;
		onGetInvitationListOk(invitelist);
		
		//正在进行--待付款
		var tobepaidlist = json.appointsGetToBePaidList;
		onGetToBePaidListOk(tobepaidlist);
		
		//正在进行--待医生确认
		var tobeconfirmeddoctorlist = json.appointsGetToBeConfirmedDoctorList;
		onGetToBeConfirmedDoctorListOk(tobeconfirmeddoctorlist);
		
		//正在进行--待病人确认（确认已就诊）
		var tobeconfirmedpatientlist = json.appointsGetToBeConfirmedPatientList;
		onGetToBeConfirmedPatientListOk(tobeconfirmedpatientlist);
		
		//正在进行--等待医生确认没有就诊
		var tobeconfirmedabsentlist = json.appointsGetToBeConfirmedUnAppointList;
		onGetToBeConfirmedPatientListOk(tobeconfirmedabsentlist);
		
		if(appointlist.length==0 && invitelist.length==0 && tobepaidlist.length==0 && tobeconfirmeddoctorlist.length==0 && tobeconfirmedpatientlist.length==0 && tobeconfirmedabsentlist.length==0) {
		    $("#szlist").hide();
		    $("#nodatadiv1").css({"display":"block","margin-top": wheight/2 - 200});
		}
		
		/*获取新邀请列表*/
		var newinvitationlist = json.appointsGetNewInvitationList;
		onGetNewInvitationListOk(newinvitationlist);
		
		//已完成 预约
		var finishedlist = json.appointsGetFinishedList;
		onGetFinishedListOk(finishedlist);
		
		//已完成 邀请
		var finishedinvitationlist = json.appointsGetFinishedInvitationList;
		onGetFinishedInvitationListOk(finishedinvitationlist);
		
		if(finishedlist.length==0 && finishedinvitationlist.length==0) {
		    $("#finishedlist").hide();
		    $("#nodatadiv3").css({"display":"block","margin-top": wheight/2 - 200});
		}
		
		//未完成 预约-退款中
		var refundinglist = json.appointsGetRefundingList;
		onGetRefundingListOk(refundinglist);
		
		//未完成 预约-已退款
		var refundedlist = json.appointsRefundedList;
		onGetRefundedListOk(refundedlist);
		
		//未完成 预约-已取消
		var acancellist = json.appointsGetACancelList;
		onGetACancelListOk(acancellist);
		
        //未完成 预约-被拒绝
        var arefusedlist = json.appointsGetARefusedList;
		onGetARefusedListOk(arefusedlist)
		
        //未完成 预约-已过期 
        var aoverduelist = json.appointsGetAOverdueList;
		onGetAOverdueListOk(aoverduelist)
		
		//未完成 邀请的各种状态
        var unfinishedinvitationlist = json.appointsGetUnfinishedInvitationList;
        onGetUnfinishedInvitationListOk(unfinishedinvitationlist)

        if(refundinglist.length==0 && refundedlist.length==0 && acancellist.length==0 && arefusedlist.length==0 && aoverduelist.length==0 && unfinishedinvitationlist.length==0) {
		    $("#unfinishedlist").hide();
		    $("#nodatadiv4").css({"display":"block","margin-top": wheight/2 - 200});
		}		
	}
}

function onGetPatientAppointListForWechatError(e) {
    $("#modalalert").modal("show");
    if(g_session.debug == true) {	
		$("#alertcontent").text("服务器出错：获取预约列表失败！"+e.responseText);
	} else {
		$("#alertcontent").text("服务器异常，请稍后重试！");
	}
}

/*获取进行中--预约列表*/
function onGetAppointListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
		for(var i=0; i<appointlist.length; i++) {
		  var doc_id = appointlist[i].doctorId;
		  var hddspatient_id = appointlist[i].hDDSPatientId;
		  var doc_photo = g_session.avatarUrl + appointlist[i].avatar;
		  var doc_name = appointlist[i].doctorName;
		  var doc_level = appointlist[i].proLevelName;
		  var doc_hospitalname = appointlist[i].hospitalName;
		  var doc_deptname = appointlist[i].deptName;
		  var appoint_startTime = appointlist[i].startTime;
		  var appoint_time = getAppointTime(appoint_startTime);
		  
		  var appoint_fee = parseFloat(appointlist[i].payFee);
		  if(appoint_fee == 0) {
			  appoint_fee = "免费";
		  } else {
			  appoint_fee += "元";
		  }
		  
			var liString = '<li class="row szinfodiv" id="appoint_item_'+hddspatient_id+'">' + 
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>' +
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>' +
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">' +
				  '<img class="img-circle" src="'+ doc_photo +'"/>' +
				'</div>' +
				'<div class="col-xs-9 col-sm-9 col-md-9">' +
					'<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">' +
						'<p><span class="sz_appoint_p">[预约]</span>'+
						'<span class="doctorname_p">'+doc_name+'</span></p>' +
						'<p class="price_p">'+appoint_fee+'</p>'+
					'</div>' +
					'<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">' +
						'<p class="appoint_info">'+ appoint_time+'</p>'+
					'</div>' +
				'</div>' +
			'</li>';
				  
			$("#appointListUl").append(liString);
			
			$("#appoint_item_" + hddspatient_id).bind("click", function() {
				var appointDoctorId = $(this).find("input[class='appointDocId']").val();
				var hDDSPatientId = $(this).find("input[class='hDDSPatientId']").val();
				localStorage.setItem("appointDoctorId", appointDoctorId); 
				localStorage.setItem("hDDSPatientId", hDDSPatientId); 
				localStorage.setItem("appointtype", 1); 
			   
				window.location.href = g_session.wxBaseUrl + "index.php?c=CIndex&m=showAppointDetail";
			});
		}
   		  
	} else {
	}
}

/*获取进行中--邀请列表*/
function onGetInvitationListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
	    for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].middleAvatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
			  
			var appoint_time = getAppointTime(appoint_startTime);
			  
			var liString = '<li class="row szinfodiv" id="invite_item_'+hddspatient_id+'">' + 
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="' + doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">' +
					    '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">' +
							'<p><span class="sz_invite_p">[邀请]</span>'+
							'<span class="doctorname_p">'+doc_name+'</span></p>' +
							'<p class="ifoverflow appoint_info">'+doc_hospitalname+'，'+doc_deptname+'</p>'+
					    '</div>' +
						'<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">' +
						    '<p class="appoint_info">'+ appoint_time+'</p>'+
					    '</div>' +
					'</div>' +
				'</li>';
			
			$("#appointListUl").append(liString);
			  
			$("#invite_item_" + hddspatient_id).bind("click", function() {
				var appointDoctorId = $(this).find("input[class='appointDocId']").val();
				var hDDSPatientId = $(this).find("input[class='hDDSPatientId']").val();
				localStorage.setItem("appointDoctorId", appointDoctorId); 
				localStorage.setItem("hDDSPatientId", hDDSPatientId); 
				localStorage.setItem("appointtype", 1); 
			   
				window.location.href = g_session.wxBaseUrl + "index.php?c=CIndex&m=showInviteDetail";
			});
		}		  
	} else {
	}
}

function onGetToBePaidListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].avatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
			var appoint_time = getAppointTime(appoint_startTime);
		  
			var appoint_fee = parseFloat(appointlist[i].payFee);
			if(appoint_fee == 0) {
				appoint_fee = "免费";
			} else {
				appoint_fee += "元";
			}
		  
			var liString = '<li class="row szinfodiv" id="tobepaid_item_'+hddspatient_id+'">' +
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
				  '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
					'<p><span class="sz_appoint_p">[预约]</span>'+
					'<span class="doctorname_p">'+doc_name+'</span></p>' +
					'<p class="price_p">'+appoint_fee+'</p>'+
				  '</div>'+
				  '<div class="col-xs-4 col-sm-4 col-md-4 appoint_status">'+
					'<input type="button" value="付款" class="btn btngreenborder"></input>'+
				  '</div>'+
				  '<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">'+
					'<p class="appoint_info">'+ appoint_time+'</p>'+
				  '</div>'+
				'</div>'+
			'</li>';
				  
			$("#appointListUl").append(liString);
			
			$("#tobepaid_item_" + hddspatient_id).bind("click", function() {
				var appointDoctorId = $(this).find("input[class='appointDocId']").val();
				var hDDSPatientId = $(this).find("input[class='hDDSPatientId']").val();
				localStorage.setItem("appointDoctorId", appointDoctorId); 
				localStorage.setItem("hDDSPatientId", hDDSPatientId); 
				localStorage.setItem("appointtype", 1); 
			   
				localStorage.setItem("toPaymentPage","appoint_payment"); 
				window.location.href = g_session.wxBaseUrl + "index.php?c=CIndex&m=showAppointPayment";
			});
		} 
					  
	} else {
	}
}

function onGetToBeConfirmedDoctorListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].avatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
		  
			var appoint_time = getAppointTime(appoint_startTime);
		  
			var liString = '<li class="row szinfodiv" id="tobeconfirmeddoctor_item_'+hddspatient_id+'">'+
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
					'<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
						'<p><span class="sz_appoint_p">[预约]</span>'+
						'<span class="doctorname_p">'+doc_name+'</span></p>' +
						'<p class="ifoverflow appoint_info">'+doc_hospitalname+'，'+doc_deptname+'</p>'+
					'</div>'+
					'<div class="col-xs-4 col-sm-4 col-md-4 appoint_status">'+
						'<p style="color:rgb(255,1,1);">待确认</p>'+
					'</div>'+
					'<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">'+
						'<p class="appoint_info">'+ appoint_time+'</p>'+
					'</div>'+
				'</div>'+
			'</li>';
				  
			$("#appointListUl").append(liString);
			
			$("#tobeconfirmeddoctor_item_" + hddspatient_id).bind("click", function() {
				var appointDoctorId = $(this).find("input[class='appointDocId']").val();
				var hDDSPatientId = $(this).find("input[class='hDDSPatientId']").val();
				localStorage.setItem("appointDoctorId", appointDoctorId); 
				localStorage.setItem("hDDSPatientId", hDDSPatientId); 
				localStorage.setItem("appointtype", 1); 
				
				localStorage.setItem("toDoctorConfirmPage","appoint_tobeconfirm");
				window.location.href = g_session.wxBaseUrl + "index.php?c=CIndex&m=showConfirmation";
			});
		}
			
	} else {
	}
}

function onGetToBeConfirmedPatientListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].avatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
			var appoint_time = getAppointTime(appoint_startTime);
			
			var appoint_fee = parseFloat(appointlist[i].payFee);
			if(appoint_fee == 0) {
				appoint_fee = "免费";
			} else {
				appoint_fee += "元";
			}
		  
			var liString = '<li class="row szinfodiv" id="tobeconfirmedpatient_item_'+hddspatient_id+'">' +
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
					'<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
						'<p><span class="sz_appoint_p">[预约]</span>'+
						'<span class="doctorname_p">'+doc_name+'</span></p>' +
						'<p class="price_p">'+appoint_fee+'</p>'+
					'</div>'+
					'<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">' +
						'<p class="appoint_info">'+ appoint_time+'</p>'+
					'</div>' +
				'</div>'+
			'</li>';
				  
			$("#appointListUl").append(liString);
			
			$("#tobeconfirmedpatient_item_" + hddspatient_id).bind("click", function() {
				var appointDoctorId = $(this).find("input[class='appointDocId']").val();
				var hDDSPatientId = $(this).find("input[class='hDDSPatientId']").val();
				localStorage.setItem("appointDoctorId", appointDoctorId); 
				localStorage.setItem("hDDSPatientId", hDDSPatientId); 
				localStorage.setItem("appointtype", 1); 
			   
				window.location.href = g_session.wxBaseUrl + "index.php?c=CIndex&m=showToBeConfirmedPatient";
			});
		}		
	} else {
	}
}

/*获取新邀请列表*/
function onGetNewInvitationListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
		$("#newinvitationListUl").html("");
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].middleAvatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
		  
			var appoint_time = getAppointTime(appoint_startTime);
		  
			var liString = '<li class="row szinfodiv" id="new_invite_item_'+hddspatient_id+'">'+
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
				  '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
						'<p><span class="sz_invite_p">[邀请]</span>'+
						'<span class="doctorname_p">'+doc_name+'</span></p>' +
						'<p class="ifoverflow appoint_info">'+doc_hospitalname+'，'+doc_deptname+'</p>'+
				  '</div>'+
				  '<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">' +
						'<p class="appoint_info">'+ appoint_time+'</p>'+
					'</div>' +
				'</div>'+
			'</li>';
				  
			$("#newinvitationListUl").append(liString);
			
			$("#new_invite_item_" + hddspatient_id).bind("click", function() {
				var appointDoctorId = $(this).find("input[class='appointDocId']").val();
				var hDDSPatientId = $(this).find("input[class='hDDSPatientId']").val();
				localStorage.setItem("appointDoctorId", appointDoctorId); 
				localStorage.setItem("hDDSPatientId", hDDSPatientId); 
				localStorage.setItem("appointtype", 2); 
			   
				window.location.href = g_session.wxBaseUrl + "index.php?c=CIndex&m=showInviteNew";
			});
		} 
		
	} else {
		$("#newinvitationlist").hide();
		$("#nodatadiv2").css({"display":"block","margin-top": wheight/2 - 200});
	}
}

/*获取已完成列表*/
//预约
function onGetFinishedListOk(appoints) {
	var appointlisttotal = appoints;
	if(appointlisttotal.length>0) {
		var appointlist = sortFinishedAppointArray(appointlisttotal);
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].avatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
			var appoint_time = getAppointTime(appoint_startTime);
			
			var appoint_fee = parseFloat(appointlist[i].payFee);
			if(appoint_fee == 0) {
				appoint_fee = "免费";
			} else {
				appoint_fee += "元";
			}
			
			var appoint_score = appointlist[i].score;
		  
			var liString = '<li class="row szinfodiv" id="finished_item_'+hddspatient_id+'">'+
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
				  '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
					'<p><span class="sz_appoint_p">[预约]</span>'+
					'<span class="doctorname_p">'+doc_name+'</span></p>' +
					'<p class="price_p">'+appoint_fee+'</p>'+
				  '</div>';
				  
			var buttonString = "";
			if(appoint_score == 0) {
				buttonString = '<div class="col-xs-4 col-sm-4 col-md-4 appoint_status">'+
					'<input type="button" value="评价" class="btn btngreenborder"></input>'+
				  '</div>';
			}
			
			var endString = '<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">'+
					'<p class="appoint_info">'+ appoint_time+'</p>'+
				  '</div>'+
				  '</div></li>';
				  
			$("#finishedListUl").append(liString + buttonString + endString);
			
			$("#finished_item_"  + hddspatient_id).bind("click", goFinishedAppointPage);
		}  
		
	} else {
	}
	
} 

//邀请
function onGetFinishedInvitationListOk(appoints) {
	var appointlisttotal = appoints;
	if(appointlisttotal.length>0) {
		var appointlist = sortFinishedAppointArray(appointlisttotal);
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].middleAvatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
		  
			var appoint_time = getAppointTime(appoint_startTime);
			var appoint_score = appointlist[i].score;
		  
			var liString = '<li class="row szinfodiv" id="finished_item_'+hddspatient_id+'">'+
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
				  '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
						'<p><span class="sz_invite_p">[邀请]</span>'+
						'<span class="doctorname_p">'+doc_name+'</span></p>' +
						'<p class="ifoverflow appoint_info">'+doc_hospitalname+'，'+doc_deptname+'</p>'+
				  '</div>';
				  
			var buttonString = "";
			if(appoint_score == 0) {
				buttonString = '<div class="col-xs-4 col-sm-4 col-md-4 appoint_status">'+
					'<input type="button" value="评价" class="btn btngreenborder"></input>'+
				  '</div>';
			}
			
			var endString = '<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">'+
					'<p class="appoint_info">'+ appoint_time+'</p>'+
				  '</div>'+
				  '</div></li>';
				  
			$("#finishedListUl").append(liString + buttonString + endString);
			$("#finished_item_"  + hddspatient_id).bind("click", goFinishedAppointPage);
		} 
			
	} else {
	}
}

/*获取未完成列表--退款中*/
function onGetRefundingListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].avatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
			var appoint_time = getAppointTime(appoint_startTime);
			
			var appoint_fee = parseFloat(appointlist[i].payFee);
			if(appoint_fee == 0) {
				appoint_fee = "免费";
			} else {
				appoint_fee += "元";
			}
		  
			var liString = '<li class="row szinfodiv refunding" id="refunding_item_'+hddspatient_id+'">'+
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
				  '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
					'<p><span class="sz_appoint_p">[预约]</span>'+
					'<span class="doctorname_p">'+doc_name+'</span></p>' +
					'<p class="price_p">'+appoint_fee+'</p>'+
				 '</div>'+
				  '<div class="col-xs-4 col-sm-4 col-md-4 appoint_status">'+
					'<p style="color:rgb(255,1,1);">退款中</p>'+
				  '</div>'+
				  '<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">'+
					'<p class="appoint_info">'+ appoint_time+'</p>'+
				  '</div>'+
				'</div>'+
			'</li>';
				  
			$("#unfinishedListUl").append(liString);
			
			$("#refunding_item_"  + hddspatient_id).bind("click", function() {
				var appointDoctorId = $(this).find("input[class='appointDocId']").val();
				var hDDSPatientId = $(this).find("input[class='hDDSPatientId']").val();
				localStorage.setItem("appointDoctorId", appointDoctorId); 
				localStorage.setItem("hDDSPatientId", hDDSPatientId); 
				localStorage.setItem("appointtype", 4); 
				
				localStorage.setItem("refundStatus","refunding");
				window.location.href = g_session.wxBaseUrl + "index.php?c=CIndex&m=showAppointRefundment";
			});
		} 

	} else {
	}
} 

/*获取未完成列表--已退款*/
function onGetRefundedListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].avatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
			var appoint_time = getAppointTime(appoint_startTime);
			
			var appoint_fee = parseFloat(appointlist[i].payFee);
			if(appoint_fee == 0) {
				appoint_fee = "免费";
			} else {
				appoint_fee += "元";
			}
		  
			var liString = '<li class="row szinfodiv refunded" id="refunded_item_'+hddspatient_id+'">'+
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
				  '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
					'<p><span class="sz_appoint_p">[预约]</span>'+
					'<span class="doctorname_p">'+doc_name+'</span></p>' +
					'<p class="price_p">'+appoint_fee+'</p>'+
				 '</div>'+
				  '<div class="col-xs-4 col-sm-4 col-md-4 appoint_status">'+
					'<p style="color:rgb(255,1,1);">已退款</p>'+
				  '</div>'+
				  '<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">'+
					'<p class="appoint_info">'+ appoint_time+'</p>'+
				  '</div>'+
				'</div>'+
			'</li>';
				  
			$("#unfinishedListUl").append(liString);
			
			$("#refunded_item_"  + hddspatient_id).bind("click", function() {
				var appointDoctorId = $(this).find("input[class='appointDocId']").val();
				var hDDSPatientId = $(this).find("input[class='hDDSPatientId']").val();
				localStorage.setItem("appointDoctorId", appointDoctorId); 
				localStorage.setItem("hDDSPatientId", hDDSPatientId); 
				localStorage.setItem("appointtype", 4); 
				
				localStorage.setItem("refundStatus","refunded");
				window.location.href = g_session.wxBaseUrl + "index.php?c=CIndex&m=showAppointRefundment";
			});
		} 

	} else {
	}
} 

/*获取未完成列表--已取消*/
function onGetACancelListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].avatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
			var appoint_time = getAppointTime(appoint_startTime);
		  
			var liString = '<li class="row szinfodiv" id="unfinished_item_'+hddspatient_id+'">'+
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
				  '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
					'<p><span class="sz_appoint_p">[预约]</span>'+
					'<span class="doctorname_p">'+doc_name+'</span></p>' +
					'<p class="ifoverflow appoint_info">'+doc_hospitalname+'，'+doc_deptname+'</p>'+
				 '</div>'+
				  '<div class="col-xs-4 col-sm-4 col-md-4 appoint_status">'+
					'<p style="color:rgb(255,1,1);">已取消</p>'+
				  '</div>'+
				  '<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">'+
					'<p class="appoint_info">'+ appoint_time+'</p>'+
				  '</div>'+
				'</div>'+
			'</li>';
	  
			$("#unfinishedListUl").append(liString);
			
			$("#unfinished_item_"  + hddspatient_id).bind("click", goUnfinishedAppointPage);
		} 

	} else {
	}
} 

/*获取未完成列表--被拒绝*/
function onGetARefusedListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].avatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
			var appoint_time = getAppointTime(appoint_startTime);
			
			var appoint_fee = parseFloat(appointlist[i].payFee);
			if(appoint_fee == 0) {
				appoint_fee = "免费";
			} else {
				appoint_fee += "元";
			}
		  
			var liString = '<li class="row szinfodiv" id="unfinished_item_'+hddspatient_id+'">'+
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
				  '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
					'<p><span class="sz_appoint_p">[预约]</span>'+
					'<span class="doctorname_p">'+doc_name+'</span></p>' +
					'<p class="ifoverflow appoint_info">'+doc_hospitalname+'，'+doc_deptname+'</p>'+
				 '</div>'+
				  '<div class="col-xs-4 col-sm-4 col-md-4 appoint_status">'+
					'<p style="color:rgb(255,1,1);">被拒绝</p>'+
				  '</div>'+
				  '<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">'+
					'<p class="appoint_info">'+ appoint_time+'</p>'+
				  '</div>'+
				'</div>'+
			'</li>';
	  
			$("#unfinishedListUl").append(liString);
			$("#unfinished_item_"  + hddspatient_id).bind("click", goUnfinishedAppointPage);
		} 

	} else {
	}
} 

/*获取未完成列表--已过期*/
function onGetAOverdueListOk(appoints) {
	var appointlist = appoints;
	if(appointlist.length>0) {
		for(var i=0; i<appointlist.length; i++) {
			var doc_id = appointlist[i].doctorId;
			var hddspatient_id = appointlist[i].hDDSPatientId;
			var doc_photo = g_session.avatarUrl + appointlist[i].avatar;
			var doc_name = appointlist[i].doctorName;
			var doc_level = appointlist[i].proLevelName;
			var doc_hospitalname = appointlist[i].hospitalName;
			var doc_deptname = appointlist[i].deptName;
			var appoint_startTime = appointlist[i].startTime;
			var appoint_time = getAppointTime(appoint_startTime);
			
			var appoint_fee = parseFloat(appointlist[i].payFee);
			if(appoint_fee == 0) {
				appoint_fee = "免费";
			} else {
				appoint_fee += "元";
			}
		  
			var liString = '<li class="row szinfodiv" id="unfinished_item_'+hddspatient_id+'">'+
				'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
				'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
				'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
				  '<img class="img-circle" src="'+ doc_photo +'"/>'+
				'</div>'+
				'<div class="col-xs-9 col-sm-9 col-md-9">'+
				  '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
					'<p><span class="sz_appoint_p">[预约]</span>'+
					'<span class="doctorname_p">'+doc_name+'</span></p>' +
					'<p class="ifoverflow appoint_info">'+doc_hospitalname+'，'+doc_deptname+'</p>'+
				 '</div>'+
				  '<div class="col-xs-4 col-sm-4 col-md-4 appoint_status">'+
					'<p style="color:rgb(255,1,1);">已过期</p>'+
				  '</div>'+
				  '<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">'+
					'<p class="appoint_info">'+ appoint_time+'</p>'+
				  '</div>'+
				'</div>'+
			'</li>';
	  
			$("#unfinishedListUl").append(liString);
			$("#unfinished_item_"  + hddspatient_id).bind("click", goUnfinishedAppointPage);
		} 

	} else {
	}
} 

/*获取未完成--邀请列表*/
function onGetUnfinishedInvitationListOk(appoints) {
	var appointlisttotal = appoints;
	if(appointlisttotal.length>0) {
	      var appointlist = sortUnfinishedInvitationArray(appointlisttotal);
	      for(var i=0; i<appointlist.length; i++) {
			  var appoint_dealtype = appointlist[i].dealType; 
			  var doc_id = appointlist[i].doctorId;
			  var hddspatient_id = appointlist[i].hDDSPatientId;
			  var doc_photo = appointlist[i].middleAvatar;
			  var doc_name = appointlist[i].doctorName;
			  var doc_level = appointlist[i].proLevelName;
			  var doc_hospitalname = appointlist[i].hospitalName;
			  var doc_deptname = appointlist[i].deptName;
			  var appoint_startTime = appointlist[i].startTime;
			  
			  var appoint_time = getAppointTime(appoint_startTime);
			  
			  var myappointstring = '<li class="row szinfodiv" id="unfinished_item_'+hddspatient_id+'">'+
					'<input type="hidden" class="appointDocId" value="'+doc_id+'"></input>'+
					'<input type="hidden" class="hDDSPatientId" value="'+hddspatient_id+'"></input>'+
					'<div class="col-xs-3 col-sm-3 col-md-3 imgdiv">'+
					  '<img class="img-circle" src="'+ doc_photo +'"/>'+
					'</div>'+
					'<div class="col-xs-9 col-sm-9 col-md-9">'+
					  '<div class="col-xs-8 col-sm-8 col-md-8 appoint_detail">'+
						'<p><span class="sz_invite_p">[邀请]</span>'+
						'<span class="doctorname_p">'+doc_name+'</span></p>' +
						'<p class="ifoverflow appoint_info">'+doc_hospitalname+'，'+doc_deptname+'</p>'+
					 '</div>'+
			         '<div class="col-xs-4 col-sm-4 col-md-4 appoint_status">';
					 
			  var dealtypedescstring;
			  if(appoint_dealtype == 11) {
				  var appoint_overdue = appointlist[i].overDue;
				  if(appoint_overdue == 1) {
					  dealtypedescstring = '<p style="color:rgb(255,1,1);">已过期</p>';  
				  } 
			  } else if(appoint_dealtype == 13||appoint_dealtype == 16) {
				  dealtypedescstring = '<p style="color:rgb(255,1,1);">已拒绝</p>';
			  } else if(appoint_dealtype == 14||appoint_dealtype == 15) {
				  dealtypedescstring = '<p style="color:rgb(255,1,1);">已取消</p>';
			  } 
		  
			  var endstring = '</div>'+
			                   '<div class="col-xs-12 col-sm-12 col-md-12 appoint_detail">'+
								'<p class="appoint_info">'+ appoint_time+'</p>'+
							  '</div>'+
			               '</div>'+
				        '</li>';
			
			  $("#unfinishedListUl").append(myappointstring + dealtypedescstring + endstring);
			  $("#unfinished_item_" + hddspatient_id).bind("click", goUnfinishedAppointPage);
		  }   
	} else {
	}
}

function sortFinishedAppointArray(appointlist) {
	var unscoredArray = new Array();//未评价
	var scoredArray = new Array();//已评价
	for(var i=0; i<appointlist.length; i++) {
		var appoint_score = appointlist[i].score; 
		if(appoint_score == 0) {
		    unscoredArray.push(appointlist[i]);
	    } else {
		    scoredArray.push(appointlist[i]);
	    } 
	}
    	 
    var a = unscoredArray.concat(scoredArray);
    return a; 	
}

function goFinishedAppointPage() {
    var appointDoctorId = $(this).find("input[class='appointDocId']").val();
	var hDDSPatientId = $(this).find("input[class='hDDSPatientId']").val();
	localStorage.setItem("appointDoctorId", appointDoctorId); 
	localStorage.setItem("hDDSPatientId", hDDSPatientId); 
	localStorage.setItem("appointtype", 3); 
   
	window.location.href = g_session.wxBaseUrl + "index.php?c=CIndex&m=showAppointFinish";
}

function sortUnfinishedInvitationArray(appointlist) {
	var refuseArray = new Array();//已拒绝
	var cancelArray = new Array();//已取消
	var overdueArray = new Array();//已过期
	for(var i=0; i<appointlist.length; i++) {
		var appoint_dealtype = appointlist[i].dealType; 
		if(appoint_dealtype == 11) {
			var appoint_overdue = appointlist[i].overDue;
			if(appoint_overdue == 1) {
				overdueArray.push(appointlist[i]);
			}
		} else if(appoint_dealtype == 13||appoint_dealtype == 16) {
			refuseArray.push(appointlist[i]);
		} else if(appoint_dealtype == 14||appoint_dealtype == 15) {
			cancelArray.push(appointlist[i]);
		}  
	}
    	
    var a = refuseArray.concat(cancelArray); 
    var b = a.concat(overdueArray);
    return b; 	
}

function goUnfinishedAppointPage() {
    var appointDoctorId = $(this).find("input[class='appointDocId']").val();
	var hDDSPatientId = $(this).find("input[class='hDDSPatientId']").val();
	localStorage.setItem("appointDoctorId", appointDoctorId); 
	localStorage.setItem("hDDSPatientId", hDDSPatientId); 
	localStorage.setItem("appointtype", 4); 
	
	window.location.href = g_session.wxBaseUrl + "index.php?c=CIndex&m=showAppointUnfinished";
}


function menuLink(type) {
	if(type == 1) {
	    $("#appointmenu").find("li").removeClass("active");
		$("#underwaylink").parent().addClass("active");
		
		$("#underway").show();
		$("#newinvitation").hide();
	    $("#unfinished").hide();
	    $("#finished").hide();
	} else if(type == 2) {
	    $("#appointmenu").find("li").removeClass("active");
		$("#newinvitationlink").parent().addClass("active");
		
		$("#newinvitation").show();
		$("#underway").hide();
		$("#unfinished").hide();
	    $("#finished").hide();
	    
	} else if(type == 3) {
	    $("#appointmenu").find("li").removeClass("active");
		$("#finishedlink").parent().addClass("active");
		
	    $("#finished").show();
		$("#underway").hide();
		$("#newinvitation").hide();
	    $("#unfinished").hide();    
	} else if(type == 4) {
	    $("#appointmenu").find("li").removeClass("active");
		$("#unfinishedlink").parent().addClass("active");
		
		$("#unfinished").show();
		$("#underway").hide();
		$("#newinvitation").hide();
	    $("#finished").hide();	    	
	}
}